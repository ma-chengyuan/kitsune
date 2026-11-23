import os
from clang.cindex import Config

if "CLANG_LIBRARY_PATH" in os.environ:
    Config.set_library_path(os.environ["CLANG_LIBRARY_PATH"])

from clang.cindex import CursorKind, TypeKind

import unittest

from .util import get_cursor, get_tu


class TestCursorKind(unittest.TestCase):
    def test_name(self):
        self.assertEqual(CursorKind.UNEXPOSED_DECL.name, "UNEXPOSED_DECL")

    def test_get_all_kinds(self):
        kinds = CursorKind.get_all_kinds()
        self.assertIn(CursorKind.UNEXPOSED_DECL, kinds)
        self.assertIn(CursorKind.TRANSLATION_UNIT, kinds)
        self.assertIn(CursorKind.VARIABLE_REF, kinds)
        self.assertIn(CursorKind.LAMBDA_EXPR, kinds)
        self.assertIn(CursorKind.OBJ_BOOL_LITERAL_EXPR, kinds)
        self.assertIn(CursorKind.OBJ_SELF_EXPR, kinds)
        self.assertIn(CursorKind.MS_ASM_STMT, kinds)
        self.assertIn(CursorKind.MODULE_IMPORT_DECL, kinds)
        self.assertIn(CursorKind.TYPE_ALIAS_TEMPLATE_DECL, kinds)

    def test_cilk_cursors(self):
        tu = get_tu(
            """
            void identity(void *);
            void reduce(void *, void *);
            int work(void);
            void update(void) {
              int cilk_reducer(identity, reduce) reducer = _Cilk_spawn work();
              _Cilk_sync;
            }
            """,
            flags=["-fopencilk", "-ftapir=none"],
        )
        self.assertEqual(list(tu.diagnostics), [])
        self.assertEqual(get_cursor(tu, "reducer").type.kind, TypeKind.HYPEROBJECT)

        cursors = {cursor.kind: cursor for cursor in tu.cursor.walk_preorder()}
        spawn_expr = cursors[CursorKind.CILK_SPAWN_EXPR]
        self.assertTrue(spawn_expr.kind.is_expression())
        self.assertFalse(spawn_expr.kind.is_statement())
        self.assertEqual(spawn_expr.type.kind, TypeKind.INT)
        self.assertIn(CursorKind.CILK_SYNC_STMT, cursors)

    def test_kind_groups(self):
        """Check that every kind classifies to exactly one group."""

        self.assertTrue(CursorKind.UNEXPOSED_DECL.is_declaration())
        self.assertTrue(CursorKind.TYPE_REF.is_reference())
        self.assertTrue(CursorKind.DECL_REF_EXPR.is_expression())
        self.assertTrue(CursorKind.UNEXPOSED_STMT.is_statement())
        self.assertTrue(CursorKind.INVALID_FILE.is_invalid())

        self.assertTrue(CursorKind.TRANSLATION_UNIT.is_translation_unit())
        self.assertFalse(CursorKind.TYPE_REF.is_translation_unit())

        self.assertTrue(CursorKind.PREPROCESSING_DIRECTIVE.is_preprocessing())
        self.assertFalse(CursorKind.TYPE_REF.is_preprocessing())

        self.assertTrue(CursorKind.UNEXPOSED_DECL.is_unexposed())
        self.assertFalse(CursorKind.TYPE_REF.is_unexposed())

        for k in CursorKind.get_all_kinds():
            group = [
                n
                for n in (
                    "is_declaration",
                    "is_reference",
                    "is_expression",
                    "is_statement",
                    "is_invalid",
                    "is_attribute",
                )
                if getattr(k, n)()
            ]

            if k in (
                CursorKind.TRANSLATION_UNIT,
                CursorKind.MACRO_DEFINITION,
                CursorKind.MACRO_INSTANTIATION,
                CursorKind.INCLUSION_DIRECTIVE,
                CursorKind.PREPROCESSING_DIRECTIVE,
                CursorKind.OVERLOAD_CANDIDATE,
            ):
                self.assertEqual(len(group), 0)
            else:
                self.assertEqual(len(group), 1)
