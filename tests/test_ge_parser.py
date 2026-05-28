from opseq.ge_parser import normalize_dtype, parse_ge_graph
from opseq.ir import InputSpec, InputDesc


def test_normalize_dtype():
    assert normalize_dtype("DT_FLOAT16") == "fp16"
    assert normalize_dtype("DT_FLOAT") == "fp32"
    assert normalize_dtype("DT_BF16") == "bf16"
    assert normalize_dtype("DT_INT64") == "int64"
    assert normalize_dtype("DT_WEIRD") == "dt_weird"  # 兜底小写


def _nodes():
    return [
        {"name": "Data_0", "type": "Data", "input_desc": [],
         "output_desc": [{"shape": [1024, 256], "dtype": "DT_FLOAT16", "format": "ND"}], "attr": {}},
        {"name": "MatMulV2_0", "type": "MatMulV2",
         "input_desc": [{"shape": [1024, 256], "dtype": "DT_FLOAT16", "format": "ND"}],
         "output_desc": [{"shape": [1024, 128], "dtype": "DT_FLOAT16", "format": "FRACTAL_NZ"}],
         "attr": {"transpose_x2": False, "fusion_group": 3}},
        {"name": "NetOutput", "type": "NetOutput", "input_desc": [], "output_desc": [], "attr": {}},
    ]


def _spec():
    return InputSpec(inputs=[InputDesc(name="dense", shape=[1024, 256], dtype="fp16")])


def test_parse_skips_structural_nodes():
    cap = parse_ge_graph(_nodes(), model_id="m", model_name="M",
                         chip="Ascend910B", input_spec=_spec())
    assert len(cap.ops) == 1
    assert cap.ops[0].op_type == "MatMul"
    assert cap.ops[0].backend_op_name == "MatMulV2_0"


def test_parse_shape_dtype_format():
    cap = parse_ge_graph(_nodes(), model_id="m", model_name="M",
                         chip="Ascend910B", input_spec=_spec())
    op = cap.ops[0]
    assert op.inputs[0].shape == [1024, 256]
    assert op.inputs[0].dtype == "fp16"
    assert op.outputs[0].format == "FRACTAL_NZ"
    assert op.fusion_group_id == 3
    assert op.attrs["transpose_x2"] is False


def test_parse_sequential_ids_and_validates():
    cap = parse_ge_graph(_nodes(), model_id="m", model_name="M",
                         chip="Ascend910B", input_spec=_spec())
    assert cap.ops[0].id == 0
    assert cap.backend == "ascend"
    cap.validate()


def test_unknown_fusion_group_defaults_to_id():
    nodes = [{"name": "Relu_0", "type": "Relu",
              "input_desc": [{"shape": [8], "dtype": "DT_FLOAT16", "format": "ND"}],
              "output_desc": [{"shape": [8], "dtype": "DT_FLOAT16", "format": "ND"}],
              "attr": {}}]
    cap = parse_ge_graph(nodes, model_id="m", model_name="M",
                         chip="Ascend910B", input_spec=_spec())
    assert cap.ops[0].fusion_group_id == 0
