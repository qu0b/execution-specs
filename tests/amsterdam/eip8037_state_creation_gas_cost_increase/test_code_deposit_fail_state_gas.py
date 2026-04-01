"""
Test block gas accounting when code deposit fails after initcode
performs state-creating operations.

When a CREATE transaction's initcode performs state-creating
operations (CALL to new account, inner CREATE) and then triggers a
code deposit failure (oversized code), the exceptional halt reverts
all state changes from the initcode execution. The state gas
consumed during initcode execution must NOT count in the block's
state_gas_used — state gas pays only for state that is actually
grown, and no state was grown on failure.

This is the #11476 variant: on top-level revert or exceptional halt,
execution_state_gas_used is reset to zero and the reservoir is
restored, consistent with child frame behavior (where
incorporate_child_on_error restores state gas to the parent).

The EELS patch for this behavior is in process_transaction: when
tx_output.error is set, state_gas_used is moved back into
state_gas_left (the reservoir) before computing tx_gas_used.

The test uses two transactions to make the delta observable:
- TX1: successful contract deployment with large code, pushing
  block_state_gas above block_regular_gas
- TX2: failing CREATE whose initcode performs a state op then
  returns oversized code

Under #11468 (state gas counted on top-level failure),
block_state_gas includes TX2's GAS_NEW_ACCOUNT. Under #11476
(this test), it does not. The difference (112 * cpsb) is
observable in header.gas_used = max(block_regular, block_state).

Tests for [EIP-8037: State Creation Gas Cost Increase]
(https://eips.ethereum.org/EIPS/eip-8037).
"""

import pytest
from execution_testing import (
    Alloc,
    Block,
    BlockchainTestFiller,
    Environment,
    Fork,
    Op,
    Transaction,
)

from .spec import ref_spec_8037

REFERENCE_SPEC_GIT_PATH = ref_spec_8037.git_path
REFERENCE_SPEC_VERSION = ref_spec_8037.version


@pytest.mark.parametrize(
    "state_op",
    [
        pytest.param(
            Op.POP(Op.CALL(gas=100_000, address=0xDEAD, value=1)),
            id="call_new_account",
        ),
        pytest.param(
            Op.POP(Op.CREATE(value=0, offset=0, size=1)),
            id="inner_create",
        ),
    ],
)
@pytest.mark.valid_from("Amsterdam")
def test_code_deposit_fail_excludes_initcode_state_gas(
    blockchain_test: BlockchainTestFiller,
    pre: Alloc,
    fork: Fork,
    state_op: bytes,
) -> None:
    """
    Verify initcode state gas excluded from block on deposit halt.

    TX1 deploys a large contract (14 KiB code) whose code deposit
    state gas pushes block_state_gas above block_regular_gas. TX2
    runs a CREATE whose initcode performs a state-creating operation
    (CALL to new account or inner CREATE, each charging
    GAS_NEW_ACCOUNT = 112 * cpsb state gas) then returns oversized
    code (> MAX_CODE_SIZE). The exceptional halt reverts initcode
    state and state gas must NOT count in block_state_gas_used,
    consistent with the principle that state gas pays only for state
    that is actually grown.

    This test validates the #11476 spec variant where top-level
    frame failure refunds all execution state gas, mirroring child
    frame behavior.
    """
    gas_limit_cap = fork.transaction_gas_limit_cap()
    assert gas_limit_cap is not None

    # --- TX1: Deploy a 14 KiB contract (high state gas) ---
    # code_deposit_state_gas = 14000 * cpsb ≈ 16.4M
    # This makes block_state_gas the binding dimension.
    deploy_size = 14_000
    tx1_create_state = fork.create_state_gas(code_size=deploy_size)

    # gas_limit above cap → surplus becomes reservoir for state gas
    tx1_gas_limit = gas_limit_cap + tx1_create_state

    sender1 = pre.fund_eoa(10**21)
    tx1 = Transaction(
        to=None,
        data=Op.RETURN(0, deploy_size),
        gas_limit=tx1_gas_limit,
        sender=sender1,
    )

    # --- TX2: Failing CREATE with initcode state ops ---
    # Initcode: state_op (charges GAS_NEW_ACCOUNT) + RETURN oversized
    oversized = fork.max_code_size() + 232  # > MAX_CODE_SIZE → halt
    initcode = state_op + Op.RETURN(0, oversized)

    sender2 = pre.fund_eoa(10**21)
    tx2 = Transaction(
        to=None,
        data=initcode,
        value=10**18,
        gas_limit=gas_limit_cap,
        sender=sender2,
    )

    blockchain_test(
        genesis_environment=Environment(gas_limit=100_000_000),
        pre=pre,
        blocks=[
            Block(
                txs=[tx1, tx2],
                gas_limit=100_000_000,
            ),
        ],
        post={},
    )
