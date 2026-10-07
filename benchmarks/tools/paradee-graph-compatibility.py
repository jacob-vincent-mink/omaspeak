# Developer-only numerical reference experiment; never a packaged runtime.
# Requires isolated onnx/onnxruntime/numpy and downloaded /tmp/paradee-survey assets.
from pathlib import Path
import json
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper
r=Path('/tmp/paradee-survey')
model=onnx.load(r/'paradee.onnx')
for i,n in enumerate(model.graph.node):
 if n.op_type.startswith('Random'):
  keep=[a for a in n.attribute if a.name!='seed']
  del n.attribute[:];n.attribute.extend(keep)
  n.attribute.append(helper.make_attribute('seed',float(i+42)))
onnx.save(model,r/'paradee_seeded_reference.onnx')
constants={}
for node in model.graph.node:
 if node.op_type=='Constant':
  for a in node.attribute:
   if a.name=='value': constants[node.output[0]]=numpy_helper.to_array(a.t)
new=[]
for i,n in enumerate(model.graph.node):
 if n.op_type=='Resize' and any(a.name=='mode' and a.s==b'linear' for a in n.attribute):
  scales=constants[n.input[2]]
  assert len(scales)==3,scales
  axes=f'compat_axes_{i}'; scale_name=f'compat_scales_{i}'
  model.graph.initializer.extend([numpy_helper.from_array(np.array([2],np.int64),axes),numpy_helper.from_array(np.array([scales[0],scales[1],1,scales[2]],np.float32),scale_name)])
  old_input=n.input[0]; old_output=n.output[0]
  n.input[0]=old_input+'_4d';n.input[2]=scale_name;n.output[0]=old_output+'_4d'
  new.extend([helper.make_node('Unsqueeze',[old_input,axes],[n.input[0]]),n,helper.make_node('Squeeze',[n.output[0],axes],[old_output])])
 else: new.append(n)
del model.graph.node[:];model.graph.node.extend(new)
onnx.checker.check_model(model)
onnx.save(model,r/'paradee_gpu4d.onnx')
# Type inference is a separate quantized-export investigation.
quant=onnx.load(r/'paradee_int8.onnx')
try:
 inferred=onnx.shape_inference.infer_shapes(quant,data_prop=True)
 onnx.save(inferred,r/'paradee_int8_inferred.onnx')
 print('INT8 inferred value types:',len(inferred.graph.value_info))
except Exception as e: print('INT8 shape inference failed:',e)
vocab=json.loads((r/'config.json').read_text())['vocab']
ids=np.array([[0]+[vocab[c] for c in 'həlˈO wˈɜɹld']+[0]],np.int64)
feed={'input_ids':ids,'speed':np.array([1],np.float32)}
so=ort.SessionOptions();so.intra_op_num_threads=1
original=ort.InferenceSession(str(r/'paradee_seeded_reference.onnx'),so,providers=['CPUExecutionProvider']).run(None,feed)[0]
changed=ort.InferenceSession(str(r/'paradee_gpu4d.onnx'),so,providers=['CPUExecutionProvider']).run(None,feed)[0]
print('rank rewrite ORT shapes:',original.shape,changed.shape)
print('rank rewrite max absolute difference:',float(np.max(np.abs(original-changed))))
