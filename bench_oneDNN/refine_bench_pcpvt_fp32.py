from cv2 import transform
import torch
import tvm
import tvm.relay as relay
import argparse
import onnx
import numpy as np

from tvm.relay.build_module import bind_params_by_name
from tvm import relay, auto_scheduler
import os

from tvm.relay.op.contrib.dnnl import pattern_table

from tvm.relay.build_module import bind_params_by_name
from tvm.relay.testing.temp_op_attr import TempOpAttr
from tvm.relay.op.contrib import dnnl

core_num = 32
os.environ["TVM_NUM_THREADS"] = str(core_num)


def detect_multi_backend(weights, input_name, imgsz):
    if 'onnx' in weights:
        onnx_model = onnx.load(weights)
        shape_dict = {input_name: (1, 3, imgsz, imgsz)}
        return relay.frontend.from_onnx(onnx_model, shape=shape_dict, dtype="float32", )
    elif 'pt' in weights:
        script_model = torch.load(weights)
        return relay.frontend.from_pytorch(script_model)


def auto_scheduling(relay_mod, relay_params, target, log_file, ntrial):
    print('Auto scheduler is extracting tasks...')
    tasks, task_weights = auto_scheduler.extract_tasks(relay_mod, relay_params, target)

    for idx, task in enumerate(tasks):
        print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
        print(task.compute_dag)
    
    tuner = auto_scheduler.TaskScheduler(tasks, task_weights , load_log_file=log_file)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=ntrial,  # change this to 20000 to achieve the best performance
        runner=auto_scheduler.LocalRunner(repeat=10, enable_cpu_cache_flush=True),
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    )   

    tuner.tune(tune_option)


@tvm.instrument.pass_instrument
class PrintIR:
    """Print the name of the pass, the IR, only before passes execute."""

    def run_before_pass(self, mod, info):
        print("Running pass: {}", info)
        print(mod)


'''
@relay.op.register_alter_op_layout("nn.conv2d", level=114)
def alter_conv2d(attrs, inputs, tinfos, out_type):
    data, weight = inputs
    def get_shape(tensor):
        if 'Var' in str(type(tensor)):
            return tensor.type_annotation.concrete_shape
        elif 'Constant' in str(type(tensor)):
            return tensor.data.shape
        elif 'TensorType' in str(type(tensor)):
            return tensor.concrete_shape
        else:
            if "pad" in tensor.op.name:
                return tensor.type_args[0].concrete_shape
            return (-1, -1, -1, -1)

    def get_dtype(tensor):
        if 'Var' in str(type(tensor)):
            return tensor.type_annotation.dtype
        elif 'Constant' in str(type(tensor)):
            return tensor.data.dtype
        elif 'TensorType' in str(type(tensor)):
            return tensor.dtype
        else:
            return 'float32'

    N, IC, IH, IW = get_shape(data)
    OC, IC, KH, KW = get_shape(weight)
    N, OC, OH, OW = get_shape(out_type)
    PH_L, PW_L, PH_R, PW_R = attrs.get_int_tuple("padding")
    SH, SW = attrs.get_int_tuple("strides")
    dilation = attrs.get_int_tuple("dilation")
    dtype = get_dtype(weight)
    G = int(attrs.groups)
    new_attrs = dict(attrs)

    if G>1: # for mobilenet
        IC = IC * G
        new_attrs['data_layout'] = "NCHW"
        new_attrs['kernel_layout'] = "OIHW"
        new_attrs['out_layout'] = "NCHW"
        return relay.nn.conv2d(data, weight, **new_attrs)

    res = relay.query_layout.AutoQuery(N,IC,KH,KW,OC,SH,SW,PH_L,PH_R,PW_L,PW_R,OH,OW,G,dtype)

    src_df, weight_df, dst_df = res.split(',')

    def trans_data(input_data, is_weight=False):
        res = ""
        if is_weight:
            trans_dic = weight_dic
        else:
            trans_dic = data_dic
        for input in input_data:
            if input.lower() in trans_dic.keys():
                trans = trans_dic[input.lower()]
                if input.islower() and input.upper() in input_data:
                    res += trans
                else:
                    res += trans.upper()
            elif input.isnumeric():
                res += input
            else:
                print("unknown char '{}' in {}".format(input, input_data))
        return res

    new_attrs['data_layout'] = trans_data(src_df, is_weight=False)
    new_attrs['kernel_layout'] = trans_data(weight_df, is_weight=True)
    new_attrs['out_layout'] = trans_data(dst_df, is_weight=False)

    if False:
        IH = OH * SH - PH_L - PH_R + KH - 1
        IW = OW * SW - PW_L - PW_R + KW - 1
        data_shape = (N, IC, IH, IW)
        print('[AlterConvLayout]-data: shape: {}, layout: old: {}, query: {}, new: {}'.format(
            data_shape, attrs['data_layout'], src_df, new_attrs['data_layout']))
        print('[AlterConvLayout]-weight: shape: {}, layout: old: {}, query: {}, new: {}'.format(
            get_shape(weight), attrs['kernel_layout'], weight_df, new_attrs['kernel_layout']))
        print('[AlterConvLayout]-out: shape: {}, layout: old: {}, query: {}, new: {}'.format(
            get_shape(out_type), attrs['out_layout'], dst_df, new_attrs['out_layout']))

    return relay.nn.conv2d(data, weight, **new_attrs)
'''

def partition_for_dnnl(mod, params=None, alter_layout=True):
    """Partition the graph greedily offloading supported operators to DNNL.

    Parameters
    ----------
    mod : Module
        The module to run passes on.
    params : Optional[Dict[str, NDArray]]
        Constant input parameters.
    Returns
    -------
    mod : Module
        Annotated and partitioned module.
    """
    if params:
        mod["main"] = bind_params_by_name(mod["main"], params)

    with TempOpAttr("nn.conv2d", "FTVMLegalize", dnnl.legalize_group_conv):
        with TempOpAttr("nn.conv2d_transpose", "FTVMLegalize", dnnl.legalize_group_conv):
            seq = tvm.transform.Sequential(
                [
                    tvm.transform.PrintIR(),
                    relay.transform.CanonicalizeOps(),
                    relay.transform.InferType(),
                    relay.transform.SimplifyInference(),
                    relay.transform.FoldConstant(),
                    relay.transform.FoldScaleAxis(),
                    # fold consecutive add ops to simplify pattern `conv2d-bias_add-bn-relu`
                    relay.transform.SimplifyExpr(),
                    relay.transform.FoldConstant(),
                    # alter group conv /conv_transpose layout to `GOIHW` / `GIOHW`
                    relay.transform.Legalize(),
                    relay.transform.FoldConstant(),
                    tvm.transform.PrintIR(),
                ]
            )
            with tvm.transform.PassContext(opt_level=3):
                mod = seq(mod)


    if alter_layout:
        with TempOpAttr("nn.conv1d", "FTVMAlterOpLayout", dnnl.alter_conv):
            with TempOpAttr("nn.conv2d", "FTVMAlterOpLayout", dnnl.alter_conv):
                with TempOpAttr("nn.conv3d", "FTVMAlterOpLayout", dnnl.alter_conv):
                    with TempOpAttr(
                        "nn.conv2d_transpose", "FTVMAlterOpLayout", dnnl.alter_conv_transpose
                    ):
                        with TempOpAttr(
                            "nn.conv3d_transpose", "FTVMAlterOpLayout", dnnl.alter_conv_transpose
                        ):
                            # with TempOpAttr(
                            #     "nn.dense", "FTVMAlterOpLayout", dnnl.alter_dense
                            # ):
                            alter_layout_seq = tvm.transform.Sequential(
                                [
                                    relay.transform.AlterOpLayout(),
                                    relay.transform.FoldConstant(),
                                ]
                            )
                            with tvm.transform.PassContext(opt_level=3):
                                mod = alter_layout_seq(mod)

    mod = dnnl.rewrite_layer_norm(mod)
    mod = dnnl.rewrite_dense_bias_gelu_reshape_last(mod, pack_wei=True)
    ###### mod = dnnl.rewrite_dense_bias_gelu_reshape_last_v2(mod, tuned_batch_size=1)
    mod = dnnl.rewrite_dense_bias_reshape_last(mod, pack_wei=True)
    mod = dnnl.rewrite_dense_to_pack(mod)

    byoc_seq = tvm.transform.Sequential(
        [
            tvm.transform.PrintIR(),
            relay.transform.MergeComposite(dnnl.pattern_table()),
            relay.transform.AnnotateTarget("dnnl"),
            relay.transform.MergeCompilerRegions(),
            relay.transform.PartitionGraph(),
            tvm.transform.PrintIR(),
        ]
    )
    with tvm.transform.PassContext(opt_level=3):
        mod = byoc_seq(mod)
        # mod = dnnl.prune_dnnl_subgraphs(mod)

    print("PrintIR:")
    print(mod)
    return mod



def main():
    args = parse_args()
    print(args)

    input_data =  np.random.uniform(-1, 1, size=(1, 3, args.imgsz, args.imgsz))

    target = tvm.target.Target("llvm -mcpu=skylake-avx512")

    relay_mod, relay_params = detect_multi_backend(args.weights, args.input_name, args.imgsz)

    relay_mod['main'] = bind_params_by_name(relay_mod['main'], relay_params)

    if args.mode == 'autosche':
        # Convert the layout from NCHW to NHWC
        desired_layouts = {'nn.conv2d': ['NHWC', 'default']}
        seq = tvm.transform.Sequential(
            [
                relay.transform.RemoveUnusedFunctions(),
                relay.transform.ConvertLayout(desired_layouts)
            ]
        )
        
        with tvm.transform.PassContext(opt_level=3):
            relay_mod = seq(relay_mod)

    elif args.mode == 'dnnlbyoc':

        relay_mod = partition_for_dnnl(relay_mod, relay_params, alter_layout=True)

        '''
        desired_layouts = {'nn.conv2d': ['NHWC', 'default']}
        seq = tvm.transform.Sequential(
            [
                tvm.transform.PrintIR(),
                relay.transform.CanonicalizeOps(),
                relay.transform.InferType(),
                relay.transform.SimplifyInference(),
                relay.transform.FoldConstant(),
                relay.transform.FoldScaleAxis(),
                # tvm.transform.PrintIR(),

                relay.transform.SimplifyExpr(),
                relay.transform.FoldConstant(),
                # tvm.transform.PrintIR(),

                relay.transform.AlterOpLayout(),
                relay.transform.ConvertLayout(desired_layouts),
                relay.transform.FoldConstant(),
                tvm.transform.PrintIR()
            ]
        )

        with tvm.transform.PassContext(opt_level=3):
            relay_mod = seq(relay_mod)

        relay_mod = rewrite_layer_norm(relay_mod)
        relay_mod = rewrite_gelu_reshape_last(relay_mod)
        
        seq1 = tvm.transform.Sequential(
            [
                tvm.transform.PrintIR(),
                relay.transform.MergeComposite(pattern_table()),
                # tvm.transform.PrintIR(),
                relay.transform.AnnotateTarget("dnnl"),
                relay.transform.MergeCompilerRegions(),
                relay.transform.PartitionGraph(),
                tvm.transform.PrintIR(),
            ]
        )

        with tvm.transform.PassContext(opt_level=3):
            relay_mod = seq1(relay_mod)
        '''
    
    
    
    if args.mode == 'autosche' and args.ntrial > 50:
        auto_scheduling(relay_mod, relay_params, target, args.tune_file, args.ntrial)
    else:
        print('skipping tuning plan...')
    
    if args.mode == 'autosche':
        with auto_scheduler.ApplyHistoryBest(args.tune_file):
            with tvm.transform.PassContext(opt_level=3, config={'relay.backend.use_auto_scheduler': True}):
                lib = relay.build(relay_mod, target=target, params=relay_params)
    elif args.mode == 'dnnlbyoc':
        print("start relay build")
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(relay_mod, target=target, params=relay_params)

    dev = tvm.device(str(target), 0)


    if args.profile:
        print('profiler enabled')
        from tvm.contrib.debugger import debug_executor as debug_graph_executor
        rt_model = debug_graph_executor.GraphModuleDebug(lib["debug_create"]("default", dev), [dev], lib.get_graph_json(), dump_root="/tmp/tvmdbg")
    else:
        from tvm.contrib import graph_executor
        rt_model = graph_executor.GraphModule(lib['default'](dev))
    

    rt_model.set_input(args.input_name, input_data)

    ## warmup loop
    for i in range(args.warmup):
        rt_model.run()

    print("warm up done.")
    
    if args.profile:
        return
    
    print("Evaluate inference time cost...")
    #ftimer = rt_model.benchmark(dev,func_name='run', repeat=args.nloop, min_repeat_ms=500)
    ftimer = rt_model.module.time_evaluator('run', dev, repeat=args.nloop)
    prof_res = np.array(ftimer().results) * 1e3  # convert to millisecond
    print("Mean inference time (std dev): %.2f ms (%.2f ms)" % (np.mean(prof_res), np.std(prof_res)))

    print("You are all set")
    


def parse_args():
    parser = argparse.ArgumentParser(description='This script is for benchmarking twins transformer inference fp32')
    parser.add_argument('--weights', type=str, help='model path')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=224, help='inference size h, w')
    parser.add_argument('--mode', type=str, help='tvm model plain/autosche/dnnlbyoc', default='plain')
    parser.add_argument('--ntrial', type=int, help='tvm auto scheduler number of trial: 20000', default=0)
    parser.add_argument('--tune-file', type=str, help='tvm auto scheduler tuning file')
    parser.add_argument('--input-name', type=str, help='onnx input name')
    parser.add_argument('--nloop', type=int, help='number of loop', default=500)
    parser.add_argument('--bs', type=int, help='number of batch size', default=1)
    parser.add_argument('--warmup', type=int, help='number of warm up loop', default=10)
    parser.add_argument('--profile', type=bool, help='print the timeline info', default=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
