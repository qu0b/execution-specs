"""
Tests for EIP-7928 Block Access Lists - OOG Gas Check Ordering.

These tests verify that account reads are NOT recorded in BAL when operations
fail due to out-of-gas. The key insight is that gas must be checked BEFORE
any account access is recorded.

Bug identified in Nethermind:
https://github.com/NethermindEth/nethermind/issues/XXXX

The bug: In Nethermind's journal implementation, AddAccountRead() is called
BEFORE checking if there's enough gas for memory expansion. This means when
OOG occurs at memory expansion, the account read entry persists in the block
access list even though the operation failed.

Correct behavior (as implemented in REVM and execution-specs):
1. Check ALL gas requirements BEFORE modifying any state
2. Only record account access after gas check succeeds
3. On OOG, no account access is recorded

The specific scenario:
1. EXTCODECOPY with a cold target address
2. Enough gas for cold account access (2600) and copy cost (3 * words)
3. NOT enough gas for memory expansion
4. The target address should NOT appear in BAL

Nethermind bug manifestation:
- AddAccountRead is called BEFORE gas check for memory expansion
- OOG occurs at memory expansion
- AddAccountRead doesn't push to journal, so it's not reverted
- Account incorrectly appears in BAL
- Block access list hash mismatch causes consensus failure
"""

import pytest
from execution_testing import (
    Account,
    Alloc,
    BalAccountExpectation,
    Block,
    BlockAccessListExpectation,
    BlockchainTestFiller,
    Bytecode,
    Fork,
    Op,
    Transaction,
)

from .spec import ref_spec_7928

REFERENCE_SPEC_GIT_PATH = ref_spec_7928.git_path
REFERENCE_SPEC_VERSION = ref_spec_7928.version


pytestmark = pytest.mark.valid_from("Amsterdam")


class TestExtcodecopyOogMemoryExpansion:
    """
    Tests for EXTCODECOPY OOG scenarios where account access should NOT
    be recorded because gas checking should happen BEFORE account access.

    This test suite catches the Nethermind bug where AddAccountRead() is
    called before checking gas for memory expansion.
    """

    def test_extcodecopy_oog_at_memory_expansion_large_offset(
        self,
        pre: Alloc,
        blockchain_test: BlockchainTestFiller,
        fork: Fork,
    ) -> None:
        """
        Test EXTCODECOPY OOG at memory expansion - account should NOT appear in BAL.

        Uses a large memory offset (64KB) to require significant memory expansion
        cost that exceeds the provided gas.

        Gas provided: push costs + cold access + copy cost (no memory expansion)
        Expected: Target should NOT appear in BAL because OOG happens before
                  the operation completes.

        Bug: Nethermind incorrectly includes target in BAL.
        """
        alice = pre.fund_eoa()
        gas_costs = fork.gas_costs()

        target_contract = pre.deploy_contract(
            code=Bytecode(Op.PUSH1(0x42) + Op.STOP)
        )

        # EXTCODECOPY with large memory offset requiring ~14000+ gas for expansion
        large_memory_offset = 0x10000  # 64KB offset
        copy_size = 32  # 1 word

        extcodecopy_contract_code = Bytecode(
            Op.PUSH1(copy_size)  # size
            + Op.PUSH1(0)  # codeOffset
            + Op.PUSH3(large_memory_offset)  # destOffset - large offset
            + Op.PUSH20(target_contract)  # address
            + Op.EXTCODECOPY  # OOG at memory expansion
            + Op.STOP
        )

        extcodecopy_contract = pre.deploy_contract(code=extcodecopy_contract_code)

        intrinsic_gas_calculator = fork.transaction_intrinsic_cost_calculator()
        intrinsic_gas_cost = intrinsic_gas_calculator()

        # Provide gas for push + cold access + copy, but NOT memory expansion
        push_cost = gas_costs.G_VERY_LOW * 4
        cold_access_cost = gas_costs.G_COLD_ACCOUNT_ACCESS
        copy_cost = gas_costs.G_COPY * 1

        tx_gas_limit = intrinsic_gas_cost + push_cost + cold_access_cost + copy_cost

        tx = Transaction(
            sender=alice,
            to=extcodecopy_contract,
            gas_limit=tx_gas_limit,
        )

        block = Block(
            txs=[tx],
            expected_block_access_list=BlockAccessListExpectation(
                account_expectations={
                    extcodecopy_contract: BalAccountExpectation.empty(),
                    # Target should NOT appear - OOG before access completes
                    # Nethermind bug: target incorrectly appears
                    target_contract: None,
                }
            ),
        )

        blockchain_test(
            pre=pre,
            blocks=[block],
            post={
                alice: Account(nonce=1),
                extcodecopy_contract: Account(),
                target_contract: Account(),
            },
        )

    def test_extcodecopy_oog_at_memory_expansion_boundary(
        self,
        pre: Alloc,
        blockchain_test: BlockchainTestFiller,
        fork: Fork,
    ) -> None:
        """
        Boundary test: EXTCODECOPY with exact gas minus 1.

        Uses a smaller memory offset for precise gas calculation, providing
        exactly 1 gas less than needed for success.

        Gas provided: push + cold access + copy + memory expansion - 1
        Expected: Target should NOT appear in BAL.

        Bug: Nethermind incorrectly includes target in BAL.
        """
        alice = pre.fund_eoa()
        gas_costs = fork.gas_costs()

        target_contract = pre.deploy_contract(code=Bytecode(Op.STOP))

        # Memory offset 256, copy 32 bytes
        # Memory cost: 9 words * 3 = 27 gas
        memory_offset = 256
        copy_size = 32

        extcodecopy_contract_code = Bytecode(
            Op.PUSH1(copy_size)
            + Op.PUSH1(0)
            + Op.PUSH2(memory_offset)
            + Op.PUSH20(target_contract)
            + Op.EXTCODECOPY
            + Op.STOP
        )

        extcodecopy_contract = pre.deploy_contract(code=extcodecopy_contract_code)

        intrinsic_gas_calculator = fork.transaction_intrinsic_cost_calculator()
        intrinsic_gas_cost = intrinsic_gas_calculator()

        push_cost = gas_costs.G_VERY_LOW * 4
        cold_access_cost = gas_costs.G_COLD_ACCOUNT_ACCESS
        copy_cost = gas_costs.G_COPY * 1
        memory_cost = 27  # 9 words * 3

        total_gas_needed = push_cost + cold_access_cost + copy_cost + memory_cost
        tx_gas_limit = intrinsic_gas_cost + total_gas_needed - 1  # Exactly 1 short

        tx = Transaction(
            sender=alice,
            to=extcodecopy_contract,
            gas_limit=tx_gas_limit,
        )

        block = Block(
            txs=[tx],
            expected_block_access_list=BlockAccessListExpectation(
                account_expectations={
                    extcodecopy_contract: BalAccountExpectation.empty(),
                    # Target should NOT appear - OOG at boundary
                    target_contract: None,
                }
            ),
        )

        blockchain_test(
            pre=pre,
            blocks=[block],
            post={
                alice: Account(nonce=1),
                extcodecopy_contract: Account(),
                target_contract: Account(),
            },
        )
