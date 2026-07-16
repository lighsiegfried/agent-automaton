"""ONNX runtime-compatibility validation (injected session factory)."""

from trainer import onnx_validate


class FakeIO:
    def __init__(self, name, shape, type="tensor(float)"):
        self.name, self.shape, self.type = name, shape, type


class FakeSession:
    def __init__(self, inputs, outputs):
        self._inputs, self._outputs = inputs, outputs

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs


def _model_file(tmp_path, data=b"onnx"):
    path = tmp_path / "m.onnx"
    path.write_bytes(data)
    return path


def _factory(inputs, outputs):
    return lambda path: FakeSession(inputs, outputs)


def test_valid_model_passes(tmp_path):
    result = onnx_validate.validate_onnx(
        _model_file(tmp_path),
        _factory([FakeIO("x", [1, 16, 96])], [FakeIO("y", [1, 1])]),
    )
    assert result["ok"] is True
    assert result["problems"] == []
    assert result["summary"]["inputs"][0]["shape"] == [1, 16, 96]


def test_missing_file_fails(tmp_path):
    result = onnx_validate.validate_onnx(tmp_path / "nope.onnx", _factory([], []))
    assert result["ok"] is False
    assert "not found" in result["problems"][0]


def test_empty_file_fails(tmp_path):
    result = onnx_validate.validate_onnx(_model_file(tmp_path, b""), _factory([], []))
    assert result["ok"] is False
    assert "empty" in result["problems"][0]


def test_unloadable_graph_fails(tmp_path):
    def boom(path):
        raise RuntimeError("bad graph")

    result = onnx_validate.validate_onnx(_model_file(tmp_path), boom)
    assert result["ok"] is False
    assert "could not load" in result["problems"][0]


def test_two_inputs_rejected(tmp_path):
    result = onnx_validate.validate_onnx(
        _model_file(tmp_path),
        _factory([FakeIO("a", [1, 16, 96]), FakeIO("b", [1, 16, 96])], [FakeIO("y", [1, 1])]),
    )
    assert result["ok"] is False
    assert any("exactly 1 input" in p for p in result["problems"])


def test_concrete_dim_mismatch_rejected(tmp_path):
    result = onnx_validate.validate_onnx(
        _model_file(tmp_path),
        _factory([FakeIO("x", [1, 8, 96])], [FakeIO("y", [1, 1])]),
    )
    assert result["ok"] is False
    assert any("trailing dims" in p for p in result["problems"])


def test_dynamic_dims_accepted(tmp_path):
    result = onnx_validate.validate_onnx(
        _model_file(tmp_path),
        _factory([FakeIO("x", ["batch", 16, 96])], [FakeIO("y", ["batch", 1])]),
    )
    assert result["ok"] is True


def test_no_outputs_rejected(tmp_path):
    result = onnx_validate.validate_onnx(
        _model_file(tmp_path), _factory([FakeIO("x", [1, 16, 96])], [])
    )
    assert result["ok"] is False
    assert any("no outputs" in p for p in result["problems"])
