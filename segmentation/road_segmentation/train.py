from __future__ import division
import numpy as np
import caffe
import re


# init
caffe.set_mode_gpu()
caffe.set_device(0)

# make a bilinear interpolation kernel
def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
 
# set parameters s.t. deconvolutional layers compute bilinear interpolation
# N.B. this is for deconvolution without groups
def interp_surgery(net, layers):
    for l in layers:
        m, k, h, w = net.params[l][0].data.shape
        if m != k:
            print 'input + output channels need to be the same'
            raise
        if h != w:
            print 'filters need to be square'
            raise
        filt = upsample_filt(h)
        net.params[l][0].data[range(m), range(k), :, :] = filt
 

step = 3000

def SolverWarper(solver_proto, base_weights):
    solver = caffe.SGDSolver(solver_proto)

    # do net surgery to set the deconvolution weights for bilinear interpolation
    interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
    interp_surgery(solver.net, interp_layers)

    # copy base weights for fine-tuning
    solver.net.copy_from(base_weights)
    
    # solve straight through -- a better approach is to define a solving loop to
    # 1. take SGD steps
    # 2. score the model by the test net `solver.test_nets[0]`
    # 3. repeat until satisfied
    solver.step(step)
    num = re.findall(r"\d+\.?\d*",solver_proto)[0]
    output_path = 'fcn' + num + '_1024_3vs3/' + 'FCN_' + num + '_iter_' + str(step) + '.caffemodel'
    solver.net.save(output_path)
    return output_path
    # output model path


output_path = SolverWarper('solverFCN32s_1024.prototxt', 'FCN_FULL_Refinement_iter_770000.caffemodel')
output_path = "./fcn32_1024_3vs3/FCN_32_iter_3000.caffemodel"
output_path = SolverWarper('solverFCN16s_1024.prototxt', output_path)
output_path = SolverWarper('solverFCN8s_1024.prototxt',  output_path)
output_path = SolverWarper('solverFCN4s_1024.prototxt',  output_path)


 
