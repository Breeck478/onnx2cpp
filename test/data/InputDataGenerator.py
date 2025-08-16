import onnx
import onnx.helper
import numpy as np
import os

modelPath = input("Enter Model path:")
outDir = os.path.join(os.path.dirname(modelPath), "test_data_set_0")
os.makedirs(outDir, exist_ok=True)
model = onnx.load(modelPath)
onnx.checker.check_model(model)
graph = model.graph

graphInputs = graph.input
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
                dims.append(dim.dim_value)  # feste Zahl
            else:
                raise ValueError("dimension may not be unkown or dynamic")
    if len(dims) == 0:
        dims.append(1) 
    np_dtype = onnx.helper.tensor_dtype_to_np_dtype(type)
    lowerBoundry = input("Enter lower boundry:")
    upperBoundry = input("Enter upper boundry:")
   
    if np.issubdtype(np_dtype, np.floating):
        floatLower = float(lowerBoundry)
        floatUpper = float(upperBoundry)
        vals = np.random.uniform(low=floatLower, high=floatUpper, size=dims).astype(np_dtype)
    elif np.issubdtype(np_dtype, np.integer):
        intLower = int(lowerBoundry)
        intUpper = int(upperBoundry)
        vals = np.random.randint(low=-intLower, high=intUpper, size=dims, dtype=np_dtype)
    else:
        raise TypeError(f"Unsupported datatype: {np_dtype}")

    testInput = onnx.helper.make_tensor(name, type, dims, vals)
    pb_bytes = testInput.SerializeToString()
    out_path = os.path.join(outDir, f"input_{idx}.pb")
    with open(out_path, "wb") as f:
        f.write(pb_bytes)
    idx += 1