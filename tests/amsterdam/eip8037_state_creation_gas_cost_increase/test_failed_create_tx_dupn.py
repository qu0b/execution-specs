"""
Test failed CREATE tx state gas with EIP-8024 DUPN in initcode.

A CREATE transaction whose initcode uses DUPN (EIP-8024, opcode 0xe6)
and fails (OOG via stack underflow or gas overflow) must not include
GAS_NEW_ACCOUNT (112 * cpsb) in block_state_gas_used. The new account
was never persisted, so its state gas must be reverted.

Regression tests for nethermind divergence at bal-devnet-3 block 589
(2026-04-08), where nethermind incorrectly charged GAS_NEW_ACCOUNT
for a failed CREATE tx, producing block gasUsed that was 112 * cpsb
= 131,488 too high.

Hive results (bal-devnet-3 images):
    geth: PASS, besu: PASS, nethermind: FAIL, reth: PASS,
    nimbus-el: PASS, ethrex: PASS

Tests for [EIP-8037: State Creation Gas Cost Increase]
(https://eips.ethereum.org/EIPS/eip-8037).
"""

import pytest
from execution_testing import (
    Account,
    Alloc,
    Block,
    BlockchainTestFiller,
    Fork,
    Op,
    Transaction,
    compute_create_address,
)

from .spec import ref_spec_8037

REFERENCE_SPEC_GIT_PATH = ref_spec_8037.git_path
REFERENCE_SPEC_VERSION = ref_spec_8037.version

# DUPN(n) = opcode 0xe6 followed by immediate byte n (EIP-8024).
# Not yet in Op helpers, so we use raw bytecode.
# PUSH32(0x200) + PUSH32(0x37) + DUPN(17): stack underflow on DUPN
# causes OOG, consuming all gas.
DUPN_OOG_INITCODE = bytes.fromhex(
    "7f0000000000000000000000000000000000000000000000000000000000000200"
    "7f0000000000000000000000000000000000000000000000000000000000000037"
    "e611"  # DUPN(17)
)

# Exact fuzz bytecode from bal-devnet-3 block 589 TX[1].
# Uses DUPN(0x01) and various ops that cause gas overflow.
BLOCK589_TX1_INITCODE = bytes.fromhex(
    "7f0000000000000000000000000000000000000000000000000000000000000200"
    "7f0000000000000000000000000000000000000000000000000000000000000037"
    "7e0101b8b5a8cce5cc1147cf5b51ab5ca3c67c50e4776e6ef8c0e0c615d924b4"
    "60def538415b5b70ac7d7a380dca8dab2c98d9fde225187a873aff5b63c541fd"
    "5a8465a6a20168e724305b5a06629724c678724ccd7b6ec672afcaceea6d29cd"
    "af5c4e2d2792fe609c582148916202ffff16555b3461008c57603f161a636b82"
    "710f5b1c095b00"
)


@pytest.mark.parametrize(
    "initcode_bytes,value",
    [
        pytest.param(
            DUPN_OOG_INITCODE,
            0,
            id="dupn_stack_underflow_oog",
        ),
        pytest.param(
            BLOCK589_TX1_INITCODE,
            0,
            id="block589_tx1_exact_bytecode",
        ),
    ],
)
@pytest.mark.valid_from("Amsterdam")
def test_failed_create_tx_no_new_account_state_gas(
    blockchain_test: BlockchainTestFiller,
    pre: Alloc,
    fork: Fork,
    initcode_bytes: bytes,
    value: int,
) -> None:
    """
    Verify failed CREATE tx does not charge new-account state gas.

    A CREATE transaction whose initcode fails (OOG, invalid opcode,
    or stack underflow) must not include GAS_NEW_ACCOUNT (112 * cpsb)
    in block_state_gas_used. The new account was never persisted, so
    its state gas is reverted.

    Nethermind bug: incorrectly includes the failed CREATE's
    new-account state gas in block_state_gas_used, producing a
    block gasUsed delta of exactly 112 * cpsb = 131,488.

    Triggered specifically by EIP-8024 opcodes (DUPN 0xe6) in the
    initcode, which cause gas overflow before the initcode completes.
    Discovered via bal-devnet-3 block 589 chain split (2026-04-08).
    """
    sender = pre.fund_eoa(10**21)

    tx = Transaction(
        to=None,
        data=initcode_bytes,
        value=value,
        gas_limit=3_000_000,
        sender=sender,
        max_priority_fee_per_gas=1,
        max_fee_per_gas=8,
    )

    created = compute_create_address(address=sender, nonce=0)

    blockchain_test(
        pre=pre,
        blocks=[Block(txs=[tx])],
        post={created: Account.NONEXISTENT},
    )


@pytest.mark.valid_from("Amsterdam")
def test_failed_create_tx_no_state_gas_block_observable(
    blockchain_test: BlockchainTestFiller,
    pre: Alloc,
    fork: Fork,
) -> None:
    """
    Observe block gasUsed after a failed CREATE tx via coinbase balance.

    TX 1: CREATE tx with EIP-8024 DUPN in initcode that OOGs. The
    failed CREATE must not contribute GAS_NEW_ACCOUNT to
    block_state_gas_used.

    TX 2: Reporter reads BALANCE(COINBASE). Any 112 * cpsb error in
    state gas accounting produces a different coinbase balance, caught
    by state root comparison against the reference implementation.

    Regression test for nethermind (bal-devnet-3 block 589).
    """
    sender = pre.fund_eoa(10**21)

    tx_create = Transaction(
        to=None,
        data=DUPN_OOG_INITCODE,
        value=0,
        gas_limit=3_000_000,
        sender=sender,
        max_priority_fee_per_gas=1,
        max_fee_per_gas=8,
    )

    reporter = pre.deploy_contract(
        code=Op.SSTORE(0, Op.BALANCE(Op.COINBASE)),
    )

    tx_reporter = Transaction(
        to=reporter,
        gas_limit=100_000,
        sender=pre.fund_eoa(),
        max_priority_fee_per_gas=1,
        max_fee_per_gas=8,
    )

    created = compute_create_address(address=sender, nonce=0)

    blockchain_test(
        pre=pre,
        blocks=[Block(txs=[tx_create, tx_reporter])],
        post={created: Account.NONEXISTENT},
    )


@pytest.mark.valid_from("Amsterdam")
def test_successful_and_failed_create_txs_in_block(
    blockchain_test: BlockchainTestFiller,
    pre: Alloc,
    fork: Fork,
) -> None:
    """
    Verify block gasUsed with successful and failed CREATE txs.

    Reproduces the exact bal-devnet-3 block 589 pattern:
    TX[0]: CREATE with value, initcode CALLs precompile then STOPs.
           Succeeds — block_state_gas_used includes GAS_NEW_ACCOUNT.
    TX[1]: CREATE with DUPN in initcode, OOGs.
           Fails — block_state_gas_used must NOT include GAS_NEW_ACCOUNT.

    The block gasUsed delta between correct and incorrect clients is
    exactly 131,488 (one GAS_NEW_ACCOUNT). The coinbase observation
    TX makes this directly observable via state root comparison.

    Regression test for nethermind (bal-devnet-3 block 589).
    """
    # TX[0]: Successful CREATE — initcode does a CALL then STOPs
    initcode_success = (
        Op.POP(
            Op.CALL(
                gas=10_000,
                address=1,  # ECRECOVER precompile
                value=0,
                args_offset=0,
                args_size=0,
                ret_offset=0,
                ret_size=0,
            )
        )
        + Op.STOP
    )

    sender_0 = pre.fund_eoa(10**21)
    tx0 = Transaction(
        to=None,
        data=bytes(initcode_success),
        value=51_255,
        gas_limit=3_000_000,
        sender=sender_0,
        max_priority_fee_per_gas=1,
        max_fee_per_gas=8,
    )

    # TX[1]: Failed CREATE — DUPN causes OOG
    sender_1 = pre.fund_eoa(10**21)
    tx1 = Transaction(
        to=None,
        data=DUPN_OOG_INITCODE,
        value=0,
        gas_limit=3_000_000,
        sender=sender_1,
        max_priority_fee_per_gas=1,
        max_fee_per_gas=8,
    )

    # TX[2]: Coinbase balance reporter
    reporter = pre.deploy_contract(
        code=Op.SSTORE(0, Op.BALANCE(Op.COINBASE)),
    )

    tx_reporter = Transaction(
        to=reporter,
        gas_limit=100_000,
        sender=pre.fund_eoa(),
        max_priority_fee_per_gas=1,
        max_fee_per_gas=8,
    )

    created_0 = compute_create_address(address=sender_0, nonce=0)
    created_1 = compute_create_address(address=sender_1, nonce=0)

    blockchain_test(
        pre=pre,
        blocks=[Block(txs=[tx0, tx1, tx_reporter])],
        post={
            created_0: Account(nonce=1, balance=51_255),
            created_1: Account.NONEXISTENT,
        },
    )
