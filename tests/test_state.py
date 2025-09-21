"""Unit tests for the AppStateStore."""

import unittest

from cam_slicer.core.state import AppState, AppStateStore


class AppStateStoreTests(unittest.TestCase):
    """Validate thread-safe state operations."""

    def setUp(self) -> None:
        self.store = AppStateStore()

    def test_update_and_read_are_isolated(self) -> None:
        """Updates should persist while reads return defensive copies."""

        updated = self.store.update(allow_execute_moves=True)
        self.assertTrue(updated.allow_execute_moves)
        snapshot = self.store.read()
        snapshot.allow_execute_moves = False
        self.assertTrue(self.store.read().allow_execute_moves)

    def test_mutate_requires_app_state(self) -> None:
        """Mutator must return an AppState instance."""

        with self.assertRaises(TypeError):
            self.store.mutate(lambda _: "invalid")

    def test_replace_and_reset(self) -> None:
        """Replacing and resetting should restore clean objects."""

        custom = AppState(allow_execute_moves=True)
        self.store.replace(custom)
        self.assertTrue(self.store.read().allow_execute_moves)
        self.store.reset()
        self.assertFalse(self.store.read().allow_execute_moves)


if __name__ == "__main__":
    unittest.main()
