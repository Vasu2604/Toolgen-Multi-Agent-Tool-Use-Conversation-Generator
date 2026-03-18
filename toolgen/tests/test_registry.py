"""
test_registry.py — Unit tests for the Tool Registry
Works with both: python -m unittest tests/test_registry.py
              and: pytest tests/test_registry.py
"""

import json, os, sys, tempfile, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from toolgen.registry import ToolRegistry

SAMPLE_TOOL = {
    "tool_name": "test_api",
    "tool_description": "A test API for unit tests",
    "category": "testing",
    "api_list": [
        {
            "name": "do_something",
            "description": "Does something useful",
            "method": "GET",
            "required_parameters": [
                {"name": "query", "type": "string", "description": "The query"},
                {"name": "limit", "type": "integer", "description": "Max results"},
            ],
            "optional_parameters": [
                {"name": "format", "type": "string", "description": "Output format",
                 "default": "json", "enum": ["json", "xml", "csv"]},
            ],
            "response_fields": ["result_id", "data", "total_count"],
        },
        {
            "name": "get_by_id",
            "description": "Get result by ID",
            "method": "GET",
            "required_parameters": [
                {"name": "result_id", "type": "string", "description": "ID"}
            ],
            "optional_parameters": [],
            "response_fields": ["result_id", "details"],
        },
    ],
}

MALFORMED_TOOL = {"description": "broken tool, no name", "api_list": []}

ALTERNATE_FORMAT_TOOL = {
    "name": "alt_api",
    "description": "Alternate field name format",
    "tool_category": "alt_category",
    "apis": [{"api_name": "alt_ep", "description": "An ep", "method": "POST",
               "required_parameters": [], "optional_parameters": []}],
}


def make_registry(*tools):
    tmpdir = tempfile.mkdtemp()
    with open(os.path.join(tmpdir, "tools.json"), "w") as f:
        json.dump(list(tools), f)
    r = ToolRegistry()
    r.load_from_directory(tmpdir)
    return r


class TestRegistryLoading(unittest.TestCase):
    def setUp(self):
        self.reg = make_registry(SAMPLE_TOOL, ALTERNATE_FORMAT_TOOL)

    def test_loads_tools(self):
        self.assertGreaterEqual(len(self.reg.get_all_tools()), 1)

    def test_tool_name(self):
        t = self.reg.get_tool("test_api")
        self.assertIsNotNone(t)
        self.assertEqual(t.name, "test_api")

    def test_tool_description(self):
        self.assertEqual(self.reg.get_tool("test_api").description, "A test API for unit tests")

    def test_tool_category(self):
        self.assertEqual(self.reg.get_tool("test_api").category, "testing")

    def test_endpoint_count(self):
        self.assertEqual(len(self.reg.get_tool("test_api").endpoints), 2)

    def test_endpoint_lookup(self):
        ep = self.reg.get_endpoint("test_api::do_something")
        self.assertIsNotNone(ep)
        self.assertEqual(ep.tool_name, "test_api")

    def test_endpoint_id_format(self):
        ep = self.reg.get_endpoint("test_api::do_something")
        self.assertEqual(ep.endpoint_id, "test_api::do_something")

    def test_required_params_present(self):
        ep = self.reg.get_endpoint("test_api::do_something")
        names = [p.name for p in ep.required_params]
        self.assertIn("query", names)
        self.assertIn("limit", names)

    def test_optional_params_present(self):
        ep = self.reg.get_endpoint("test_api::do_something")
        names = [p.name for p in ep.optional_params]
        self.assertIn("format", names)

    def test_param_enum(self):
        ep = self.reg.get_endpoint("test_api::do_something")
        fmt = next(p for p in ep.optional_params if p.name == "format")
        self.assertIn("json", fmt.enum)
        self.assertIn("xml", fmt.enum)

    def test_param_default(self):
        ep = self.reg.get_endpoint("test_api::do_something")
        fmt = next(p for p in ep.optional_params if p.name == "format")
        self.assertEqual(fmt.default, "json")

    def test_response_fields(self):
        ep = self.reg.get_endpoint("test_api::do_something")
        self.assertIn("result_id", ep.response_fields)

    def test_required_flag_true(self):
        ep = self.reg.get_endpoint("test_api::do_something")
        for p in ep.required_params:
            self.assertTrue(p.required)

    def test_optional_flag_false(self):
        ep = self.reg.get_endpoint("test_api::do_something")
        for p in ep.optional_params:
            self.assertFalse(p.required)

    def test_alternate_field_names_loaded(self):
        self.assertIsNotNone(self.reg.get_tool("alt_api"))


class TestMalformedHandling(unittest.TestCase):
    def test_skips_tool_without_name(self):
        reg = make_registry(MALFORMED_TOOL, SAMPLE_TOOL)
        self.assertIsNotNone(reg.get_tool("test_api"))

    def test_nonexistent_tool_returns_none(self):
        reg = make_registry(SAMPLE_TOOL)
        self.assertIsNone(reg.get_tool("nonexistent"))

    def test_nonexistent_endpoint_returns_none(self):
        reg = make_registry(SAMPLE_TOOL)
        self.assertIsNone(reg.get_endpoint("fake::ep"))

    def test_empty_directory(self):
        tmpdir = tempfile.mkdtemp()
        reg = ToolRegistry()
        count = reg.load_from_directory(tmpdir)
        self.assertEqual(count, 0)


class TestRegistrySave(unittest.TestCase):
    def test_save_valid_json(self):
        reg = make_registry(SAMPLE_TOOL)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            reg.save(path)
            data = json.load(open(path))
            self.assertIsInstance(data, list)
            self.assertGreater(len(data), 0)
            self.assertIn("endpoints", data[0])
        finally:
            os.unlink(path)

    def test_summary_keys(self):
        reg = make_registry(SAMPLE_TOOL)
        s = reg.summary()
        for k in ["total_tools", "total_endpoints", "categories"]:
            self.assertIn(k, s)

    def test_summary_counts(self):
        reg = make_registry(SAMPLE_TOOL)
        s = reg.summary()
        self.assertEqual(s["total_tools"], 1)
        self.assertEqual(s["total_endpoints"], 2)


if __name__ == "__main__":
    unittest.main()
