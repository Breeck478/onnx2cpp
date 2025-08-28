import onnx
import onnx.helper
import onnxruntime as ort
import numpy as np
import os
from onnx import version_converter

modelPath = input("Enter Model path:")
outDir = os.path.join(os.path.dirname(modelPath), "test_data_set_0")
os.makedirs(outDir, exist_ok=True)
model = onnx.load(modelPath)
# Downgrade Model because onnxruntime does not accept ir version 11
#if (model.ir_version > 10):
model.ir_version = 10
model = version_converter.convert_version(model, 22)

onnx.save(model, modelPath)
model = onnx.load(modelPath)

onnx.checker.check_model(model)
graph = model.graph

graphInputs = graph.input
inputDict = {}
idx = 0
for graphInput in graphInputs:
    
    name = graphInput.name
    print("Generating Data for " + name + "-input") 
    tensor_type = graphInput.type.tensor_type
    type = tensor_type.elem_type
    dims = []
    if tensor_type.HasField("shape"):
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                dims.append(dim.dim_value) 
            else:
                dims.append(int(input("Enter Dimension for dynmic Dimension")))
    if len(dims) == 0:
        dims.append(1) 
    np_dtype = onnx.helper.tensor_dtype_to_np_dtype(type)
    lowerBoundry = int(input("Enter lower boundry:"))
    upperBoundry = int(input("Enter upper boundry:"))
   
    if np.issubdtype(np_dtype, np.float32):
        floatLower = float(lowerBoundry)
        floatUpper = float(upperBoundry)
        vals = np.random.uniform(low=floatLower, high=floatUpper, size=dims).astype(np_dtype)
    elif np.issubdtype(np_dtype, np.integer):
        intLower = int(lowerBoundry)
        intUpper = int(upperBoundry)
        vals = np.random.randint(low=-intLower, high=intUpper, size=dims, dtype=np_dtype).astype(np_dtype)
    else:
        raise TypeError(f"Unsupported datatype: {np_dtype}")


    


    testInput = onnx.helper.make_tensor(name, type, dims, vals)
    inputDict[name] = vals
    pb_bytes = testInput.SerializeToString()
    out_path = os.path.join(outDir, f"input_{idx}.pb")
    with open(out_path, "wb") as f:
        f.write(pb_bytes)
    idx += 1

sess = ort.InferenceSession(modelPath)
outputs = sess.run(None, inputDict)
idx = 0
for output in outputs:
    name =  model.graph.output[idx].name
    np_dtype = output.dtype
    dims = output.shape if hasattr(output, 'shape') else [1]
    if not hasattr(output, 'shape'):# if output is scalar, convert it into tensor Does not work correctly. Therefore make sure the ouputs has at least 2 dimensions
        output = np.array([output]) 
    testInput = onnx.helper.make_tensor(name, onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np_dtype], dims, output)
    pb_bytes = testInput.SerializeToString()
    out_path = os.path.join(outDir, f"output_{idx}.pb")
    with open(out_path, "wb") as f:
        f.write(pb_bytes)
    idx += 1