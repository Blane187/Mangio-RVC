import sys
from shutil import rmtree
import json  # Mangio fork using json for preset saving


from glob import glob1
from signal import SIGTERM
import os

now_dir = os.getcwd()
sys.path.append(now_dir)

from LazyImport import lazyload

math = lazyload("math")

import traceback
import warnings

tensorlowest = lazyload("tensorlowest")
import faiss

ffmpeg = lazyload("ffmpeg")

np = lazyload("numpy")
torch = lazyload("torch")
re = lazyload("regex")
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["no_proxy"] = "localhost, 127.0.0.1, ::1"
import logging
from random import shuffle
from subprocess import Popen

gr = lazyload("gradio")
SF = lazyload("soundfile")
SFWrite = SF.write
from config import Config
from fairseq import checkpoint_utils
from i18n import I18nAuto
from lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from lib.infer_pack.models_onnx import SynthesizerTrnMsNSFsidM
from infer_uvr5 import _audio_pre_, _audio_pre_new
from MDXNet import MDXNetDereverb
from my_utils import load_audio, CSVutil
from train.process_ckpt import change_info, extract_small_model, merge, show_info
from vc_infer_pipeline import VC
from sklearn.cluster import MiniBatchKMeans

import time
import threading

from numba import jit

from shlex import quote as SQuote

RQuote = lambda val: SQuote(str(val))

tmp = os.path.join(now_dir, "TEMP")
runtime_dir = os.path.join(now_dir, "runtime/Lib/site-packages")
directories = ["logs", "audios", "datasets", "weights"]

rmtree(tmp, ignore_errors=True)
rmtree(os.path.join(runtime_dir, "infer_pack"), ignore_errors=True)
rmtree(os.path.join(runtime_dir, "uvr5_pack"), ignore_errors=True)

os.makedirs(tmp, exist_ok=True)
for folder in directories:
    os.makedirs(os.path.join(now_dir, folder), exist_ok=True)

os.environ["TEMP"] = tmp
warnings.filterwarnings("ignore")
torch.manual_seed(114514)
logging.getLogger("numba").setLevel(logging.WARNING)

os.makedirs("csvdb/", exist_ok=True)
with open("csvdb/formanting.csv", "a"):
    pass
with open("csvdb/stop.csv", "a"):
    pass

global DoFormant, Quefrency, Timbre

try:
    DoFormant, Quefrency, Timbre = CSVutil("csvdb/formanting.csv", "r", "formanting")
    DoFormant = DoFormant.lower() == "true"
except (ValueError, TypeError, IndexError):
    DoFormant, Quefrency, Timbre = False, 1.0, 1.0
    CSVutil("csvdb/formanting.csv", "w+", "formanting", DoFormant, Quefrency, Timbre)

config = Config()
i18n = I18nAuto()
i18n.print()
# 判断是否有能用来训练和加速推理的N卡
ngpu = torch.cuda.device_count()
gpu_infos = []
mem = []
if_gpu_ok = False

keywords = [
    "10",
    "16",
    "20",
    "30",
    "40",
    "A2",
    "A3",
    "A4",
    "P4",
    "A50",
    "500",
    "A60",
    "70",
    "80",
    "90",
    "M4",
    "T4",
    "TITAN",
]

if torch.cuda.is_available() or ngpu != 0:
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i).upper()
        if any(keyword in gpu_name for keyword in keywords):
            if_gpu_ok = True  # 至少有一张能用的N卡
            gpu_infos.append("%s\t%s" % (i, gpu_name))
            mem.append(
                int(torch.cuda.get_device_properties(i).total_memory / 1e9 + 0.4)
            )

gpu_info = (
    "\n".join(gpu_infos) if if_gpu_ok and gpu_infos else i18n("很遗憾您这没有能用的显卡来支持您训练")
)
default_batch_size = min(mem) // 2 if if_gpu_ok and gpu_infos else 1
gpus = "-".join(i[0] for i in gpu_infos)

hubert_model = None


def load_hubert():
    global hubert_model
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        ["hubert_base.pt"], suffix=""
    )
    hubert_model = models[0].to(config.device)

    if config.is_half:
        hubert_model = hubert_model.half()

    hubert_model.eval()


weight_root = "weights"
weight_uvr5_root = "uvr5_weights"
index_root = "./logs/"
audio_root = "audios"

names = [name for name in os.listdir(weight_root) if name.endswith((".pth", ".onnx"))]

indexes_list = [
    f"{root}/{name}"
    for root, _, files in os.walk(index_root, topdown=False)
    for name in files
    if name.endswith(".index") and "trained" not in name
]

audio_paths = [
    f"{root}/{name}"
    for root, _, files in os.walk(audio_root, topdown=False)
    for name in files
]

uvr5_names = [
    name.replace(".pth", "")
    for name in os.listdir(weight_uvr5_root)
    if name.endswith(".pth") or "onnx" in name
]

check_for_name = lambda: sorted(names)[0] if names else ""


def get_indexes():
    indexes_list = [
        os.path.join(dirpath, filename).replace("\\", "/")
        for dirpath, _, filenames in os.walk("./logs/")
        for filename in filenames
        if filename.endswith(".index") and "trained" not in filename
    ]

    return indexes_list if indexes_list else ""


def get_fshift_presets():
    fshift_presets_list = [
        os.path.join(dirpath, filename).replace("\\", "/")
        for dirpath, _, filenames in os.walk("./formantshiftcfg/")
        for filename in filenames
        if filename.endswith(".txt")
    ]

    return fshift_presets_list if fshift_presets_list else ""


def vc_single(
    sid,
    input_audio_path0,
    input_audio_path1,
    f0_up_key,
    f0_file,
    f0_method,
    file_index,
    file_index2,
    # file_big_npy,
    index_rate,
    filter_radius,
    resample_sr,
    rms_mix_rate,
    protect,
    crepe_hop_length,
):
    global tgt_sr, net_g, vc, hubert_model, version
    if not input_audio_path0 and not input_audio_path1:
        return "You need to upload an audio", None

    f0_up_key = int(f0_up_key)

    try:
        reliable_path = (
            input_audio_path1 if input_audio_path0 == "" else input_audio_path0
        )
        audio = load_audio(reliable_path, 16000, DoFormant, Quefrency, Timbre)

        audio_max = np.abs(audio).max() / 0.95
        if audio_max > 1:
            audio /= audio_max

        times = [0, 0, 0]
        if not hubert_model:
            load_hubert()

        if_f0 = cpt.get("f0", 1)
        file_index = (
            (
                file_index.strip(" ")
                .strip('"')
                .strip("\n")
                .strip('"')
                .strip(" ")
                .replace("trained", "added")
            )
            if file_index != ""
            else file_index2
        )

        audio_opt = vc.pipeline(
            hubert_model,
            net_g,
            sid,
            audio,
            input_audio_path1,
            times,
            f0_up_key,
            f0_method,
            file_index,
            index_rate,
            if_f0,
            filter_radius,
            tgt_sr,
            resample_sr,
            rms_mix_rate,
            version,
            protect,
            crepe_hop_length,
            f0_file=f0_file,
        )

        if tgt_sr != resample_sr >= 16000:
            tgt_sr = resample_sr

        index_info = (
            "Using index:%s." % file_index
            if os.path.exists(file_index)
            else "Index not used."
        )

        return (
            f"Success.\n {index_info}\nTime:\n npy:{times[0]}, f0:{times[1]}, infer:{times[2]}",
            (tgt_sr, audio_opt),
        )
    except:
        info = traceback.format_exc()
        print(info)
        return info, (None, None)


def vc_multi(
    sid,
    dir_path,
    opt_root,
    paths,
    f0_up_key,
    f0_method,
    file_index,
    file_index2,
    # file_big_npy,
    index_rate,
    filter_radius,
    resample_sr,
    rms_mix_rate,
    protect,
    format1,
    crepe_hop_length,
):
    try:
        dir_path, opt_root = [
            x.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
            for x in [dir_path, opt_root]
        ]
        os.makedirs(opt_root, exist_ok=True)
        paths = (
            [os.path.join(dir_path, name) for name in os.listdir(dir_path)]
            if dir_path
            else [path.name for path in paths]
        )
        infos = []

        for path in paths:
            info, opt = vc_single(
                sid,
                path,
                None,
                f0_up_key,
                None,
                f0_method,
                file_index,
                file_index2,
                index_rate,
                filter_radius,
                resample_sr,
                rms_mix_rate,
                protect,
                crepe_hop_length,
            )
            if "Success" in info:
                try:
                    tgt_sr, audio_opt = opt
                    output_path = f"{opt_root}/{os.path.basename(path)}"
                    path, extension = (
                        output_path
                        if format1 in ["wav", "flac", "mp3", "ogg", "aac"]
                        else f"{output_path}.wav",
                        format1,
                    )
                    SFWrite(path, audio_opt, tgt_sr)
                    if os.path.exists(path) and extension not in [
                        "wav",
                        "flac",
                        "mp3",
                        "ogg",
                        "aac",
                    ]:
                        os.system(
                            f"ffmpeg -i {RQuote(path)} -vn {RQuote(path[:-4])}.{RQuote(extension)} -q:a 2 -y"
                        )
                except:
                    info += traceback.format_exc()
            infos.append(f"{os.path.basename(path)}->{info}")
            yield "\n".join(infos)
        yield "\n".join(infos)
    except:
        yield traceback.format_exc()


def uvr(model_name, inp_root, save_root_vocal, paths, save_root_ins, agg, format0):
    infos = []
    try:
        inp_root, save_root_vocal, save_root_ins = [
            x.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
            for x in [inp_root, save_root_vocal, save_root_ins]
        ]

        pre_fun = (
            MDXNetDereverb(15)
            if model_name == "onnx_dereverb_By_FoxJoy"
            else (_audio_pre_ if "DeEcho" not in model_name else _audio_pre_new)(
                agg=int(agg),
                model_path=os.path.join(weight_uvr5_root, model_name + ".pth"),
                device=config.device,
                is_half=config.is_half,
            )
        )

        paths = (
            [os.path.join(inp_root, name) for name in os.listdir(inp_root)]
            if inp_root
            else [path.name for path in paths]
        )

        for path in paths:
            inp_path = os.path.join(inp_root, path)
            need_reformat, done = 1, 0

            try:
                info = ffmpeg.probe(inp_path, cmd="ffprobe")
                if (
                    info["streams"][0]["channels"] == 2
                    and info["streams"][0]["sample_rate"] == "44100"
                ):
                    need_reformat = 0
                    pre_fun._path_audio_(
                        inp_path, save_root_ins, save_root_vocal, format0
                    )
                    done = 1
            except:
                traceback.print_exc()

            if need_reformat:
                tmp_path = f"{tmp}/{os.path.basename(RQuote(inp_path))}.reformatted.wav"
                os.system(
                    f"ffmpeg -i {RQuote(inp_path)} -vn -acodec pcm_s16le -ac 2 -ar 44100 {RQuote(tmp_path)} -y"
                )
                inp_path = tmp_path

            try:
                if not done:
                    pre_fun._path_audio_(
                        inp_path, save_root_ins, save_root_vocal, format0
                    )
                infos.append(f"{os.path.basename(inp_path)}->Success")
                yield "\n".join(infos)
            except:
                infos.append(f"{os.path.basename(inp_path)}->{traceback.format_exc()}")
                yield "\n".join(infos)
    except:
        infos.append(traceback.format_exc())
        yield "\n".join(infos)
    finally:
        try:
            if model_name == "onnx_dereverb_By_FoxJoy":
                del pre_fun.pred.model
                del pre_fun.pred.model_
            else:
                del pre_fun.model

            del pre_fun
        except:
            traceback.print_exc()

        print("clean_empty_cache")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    yield "\n".join(infos)


def get_vc(sid, to_return_protect0, to_return_protect1):
    global n_spk, tgt_sr, net_g, vc, cpt, version, hubert_model
    if not sid:
        if hubert_model is not None:
            print("clean_empty_cache")
            del net_g, n_spk, vc, hubert_model, tgt_sr
            hubert_model = net_g = n_spk = vc = hubert_model = tgt_sr = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if_f0, version = cpt.get("f0", 1), cpt.get("version", "v1")
            net_g = (
                (
                    SynthesizerTrnMs256NSFsid
                    if version == "v1"
                    else SynthesizerTrnMs768NSFsid
                )(*cpt["config"], is_half=config.is_half)
                if if_f0 == 1
                else (
                    SynthesizerTrnMs256NSFsid_nono
                    if version == "v1"
                    else SynthesizerTrnMs768NSFsid_nono
                )(*cpt["config"])
            )
            del net_g, cpt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            cpt = None
        return ({"visible": False, "__type__": "update"},) * 3

    person = f"{weight_root}/{sid}"
    print(f"loading {person}")
    cpt = torch.load(person, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]

    if cpt.get("f0", 1) == 0:
        to_return_protect0 = to_return_protect1 = {
            "visible": False,
            "value": 0.5,
            "__type__": "update",
        }
    else:
        to_return_protect0 = {
            "visible": True,
            "value": to_return_protect0,
            "__type__": "update",
        }
        to_return_protect1 = {
            "visible": True,
            "value": to_return_protect1,
            "__type__": "update",
        }

    version = cpt.get("version", "v1")
    net_g = (
        (SynthesizerTrnMs256NSFsid if version == "v1" else SynthesizerTrnMs768NSFsid)(
            *cpt["config"], is_half=config.is_half
        )
        if cpt.get("f0", 1) == 1
        else (
            SynthesizerTrnMs256NSFsid_nono
            if version == "v1"
            else SynthesizerTrnMs768NSFsid_nono
        )(*cpt["config"])
    )
    del net_g.enc_q

    print(net_g.load_state_dict(cpt["weight"], strict=False))
    net_g.eval().to(config.device)
    net_g = net_g.half() if config.is_half else net_g.float()

    vc = VC(tgt_sr, config)
    n_spk = cpt["config"][-3]

    return (
        {"visible": True, "maximum": n_spk, "__type__": "update"},
        to_return_protect0,
        to_return_protect1,
    )


def change_choices():
    names = [name for name in os.listdir(weight_root) if name.endswith(".pth", ".onnx")]
    indexes_list = [
        os.path.join(root, name)
        for root, _, files in os.walk(index_root, topdown=False)
        for name in files
        if name.endswith(".index") and "trained" not in name
    ]
    audio_paths = [
        os.path.join(audio_root, file)
        for file in os.listdir(os.path.join(os.getcwd(), "audios"))
    ]

    return (
        {"choices": sorted(names), "__type__": "update"},
        {"choices": sorted(indexes_list), "__type__": "update"},
        {"choices": sorted(audio_paths), "__type__": "update"},
    )


sr_dict = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}


@jit
def if_done(done, p):
    while p.poll() is None:
        time.sleep(0.5)

    done[0] = True


def if_done_multi(done, ps):
    while not all(p.poll() is not None for p in ps):
        time.sleep(0.5)
    done[0] = True


def formant_enabled(
    cbox, qfrency, tmbre, frmntapply, formantpreset, formant_refresh_button
):
    global DoFormant
    DoFormant = cbox

    CSVutil("csvdb/formanting.csv", "w+", "formanting", DoFormant, qfrency, tmbre)
    visibility_update = {"visible": DoFormant, "__type__": "update"}

    return ({"value": DoFormant, "__type__": "update"},) + (visibility_update,) * 6


def formant_apply(qfrency, tmbre):
    global Quefrency, Timbre, DoFormant

    Quefrency = qfrency
    Timbre = tmbre
    DoFormant = True

    CSVutil("csvdb/formanting.csv", "w+", "formanting", DoFormant, Quefrency, Timbre)

    return (
        {"value": Quefrency, "__type__": "update"},
        {"value": Timbre, "__type__": "update"},
    )


def update_fshift_presets(preset, qfrency, tmbre):
    if preset:
        with open(preset, "r") as p:
            content = p.readlines()
            qfrency, tmbre = content[0].strip(), content[1]

        formant_apply(qfrency, tmbre)
    else:
        qfrency, tmbre = preset_apply(preset, qfrency, tmbre)

    return (
        {"choices": get_fshift_presets(), "__type__": "update"},
        {"value": qfrency, "__type__": "update"},
        {"value": tmbre, "__type__": "update"},
    )


def preprocess_dataset(trainset_dir, exp_dir, sr, n_p):
    sr = sr_dict[sr]

    log_dir = os.path.join(now_dir, "logs", exp_dir)
    log_file = os.path.join(log_dir, "preprocess.log")

    os.makedirs(log_dir, exist_ok=True)

    with open(log_file, "w") as f:
        pass

    cmd = (
        f"{config.python_cmd} "
        "trainset_preprocess_pipeline_print.py "
        f"{trainset_dir} "
        f"{RQuote(sr)} "
        f"{RQuote(n_p)} "
        f"{log_dir} "
        f"{RQuote(config.noparallel)}"
    )
    print(cmd)

    p = Popen(cmd, shell=True)
    done = [False]

    threading.Thread(
        target=if_done,
        args=(
            done,
            p,
        ),
    ).start()

    while not done[0]:
        with open(log_file, "r") as f:
            yield f.read()
        time.sleep(1)

    with open(log_file, "r") as f:
        log = f.read()

    print(log)
    yield log


def extract_f0_feature(gpus, n_p, f0method, if_f0, exp_dir, version19, echl):
    gpus = gpus.split("-")
    log_dir = f"{now_dir}/logs/{exp_dir}"
    log_file = f"{log_dir}/extract_f0_feature.log"
    os.makedirs(log_dir, exist_ok=True)
    with open(log_file, "w") as f:
        pass

    if if_f0:
        cmd = (
            f"{config.python_cmd} extract_f0_print.py {log_dir} "
            f"{RQuote(n_p)} {RQuote(f0method)} {RQuote(echl)}"
        )
        print(cmd)
        p = Popen(cmd, shell=True, cwd=now_dir)
        done = [False]
        threading.Thread(target=if_done, args=(done, p)).start()

        while not done[0]:
            with open(log_file, "r") as f:
                yield f.read()
            time.sleep(1)

    leng = len(gpus)
    ps = []

    for idx, n_g in enumerate(gpus):
        cmd = (
            f"{config.python_cmd} extract_feature_print.py {RQuote(config.device)} "
            f"{RQuote(leng)} {RQuote(idx)} {RQuote(n_g)} {log_dir} {RQuote(version19)}"
        )
        print(cmd)
        p = Popen(cmd, shell=True, cwd=now_dir)
        ps.append(p)

    done = [False]
    threading.Thread(target=if_done_multi, args=(done, ps)).start()

    while not done[0]:
        with open(log_file, "r") as f:
            yield f.read()
        time.sleep(1)

    with open(log_file, "r") as f:
        log = f.read()

    print(log)
    yield log


def change_sr2(sr2, if_f0_3, version19):
    path_str = "" if version19 == "v1" else "_v2"
    f0_str = "f0" if if_f0_3 else ""
    model_paths = {"G": "", "D": ""}

    for model_type in model_paths:
        file_path = f"pretrained{path_str}/{f0_str}{model_type}{sr2}.pth"
        if os.access(file_path, os.F_OK):
            model_paths[model_type] = file_path
        else:
            print(f"{file_path} doesn't exist, will not use pretrained model.")

    return (model_paths["G"], model_paths["D"])


def change_version19(sr2, if_f0_3, version19):
    path_str = "" if version19 == "v1" else "_v2"
    sr2 = "40k" if (sr2 == "32k" and version19 == "v1") else sr2
    choices_update = (
        {"choices": ["40k", "48k"], "__type__": "update", "value": sr2}
        if version19 == "v1"
        else {"choices": ["40k", "48k", "32k"], "__type__": "update", "value": sr2}
    )

    f0_str = "f0" if if_f0_3 else ""
    model_paths = {"G": "", "D": ""}

    for model_type in model_paths:
        file_path = f"pretrained{path_str}/{f0_str}{model_type}{sr2}.pth"
        if os.access(file_path, os.F_OK):
            model_paths[model_type] = file_path
        else:
            print(f"{file_path} doesn't exist, will not use pretrained model.")

    return (model_paths["G"], model_paths["D"], choices_update)


def change_f0(if_f0_3, sr2, version19):  # f0method8,pretrained_G14,pretrained_D15
    path_str = "" if version19 == "v1" else "_v2"

    pth_format = "pretrained%s/f0%s%s.pth"
    model_desc = {"G": "", "D": ""}

    for model_type in model_desc:
        file_path = pth_format % (path_str, model_type, sr2)
        if os.access(file_path, os.F_OK):
            model_desc[model_type] = file_path
        else:
            print(file_path, "doesn't exist, will not use pretrained model")

    return (
        {"visible": if_f0_3, "__type__": "update"},
        model_desc["G"],
        model_desc["D"],
        {"visible": if_f0_3, "__type__": "update"},
    )


global log_interval


def set_log_interval(exp_dir, batch_size12):
    log_interval = 1
    folder_path = os.path.join(exp_dir, "1_16k_wavs")

    if os.path.isdir(folder_path):
        wav_files_num = len(glob1(folder_path, "*.wav"))

        if wav_files_num > 0:
            log_interval = math.ceil(wav_files_num / batch_size12)
            if log_interval > 1:
                log_interval += 1

    return log_interval


def click_train(
    exp_dir1,
    sr2,
    if_f0_3,
    spk_id5,
    save_epoch10,
    total_epoch11,
    batch_size12,
    if_save_latest13,
    pretrained_G14,
    pretrained_D15,
    gpus16,
    if_cache_gpu17,
    if_save_every_weights18,
    version19,
):
    CSVutil("csvdb/stop.csv", "w+", "formanting", False)

    log_dir = os.path.join(now_dir, "logs", exp_dir1)

    os.makedirs(log_dir, exist_ok=True)

    gt_wavs_dir = os.path.join(log_dir, "0_gt_wavs")
    feature_dim = "256" if version19 == "v1" else "768"

    feature_dir = os.path.join(log_dir, f"3_feature{feature_dim}")

    log_interval = set_log_interval(log_dir, batch_size12)

    required_dirs = [gt_wavs_dir, feature_dir]

    if if_f0_3:
        f0_dir = f"{log_dir}/2a_f0"
        f0nsf_dir = f"{log_dir}/2b-f0nsf"
        required_dirs.extend([f0_dir, f0nsf_dir])

    names = set(
        name.split(".")[0]
        for directory in required_dirs
        for name in os.listdir(directory)
    )

    def generate_paths(name):
        paths = [gt_wavs_dir, feature_dir]
        if if_f0_3:
            paths.extend([f0_dir, f0nsf_dir])
        return "|".join(
            [
                path.replace("\\", "\\\\")
                + "/"
                + name
                + (
                    ".wav.npy"
                    if path in [f0_dir, f0nsf_dir]
                    else ".wav"
                    if path == gt_wavs_dir
                    else ".npy"
                )
                for path in paths
            ]
        )

    opt = [f"{generate_paths(name)}|{spk_id5}" for name in names]
    mute_dir = f"{now_dir}/logs/mute"

    for _ in range(2):
        mute_string = f"{mute_dir}/0_gt_wavs/mute{sr2}.wav|{mute_dir}/3_feature{feature_dim}/mute.npy"
        if if_f0_3:
            mute_string += (
                f"|{mute_dir}/2a_f0/mute.wav.npy|{mute_dir}/2b-f0nsf/mute.wav.npy"
            )
        opt.append(mute_string + f"|{spk_id5}")

    shuffle(opt)
    with open(f"{log_dir}/filelist.txt", "w") as f:
        f.write("\n".join(opt))

    print("write filelist done")
    print("use gpus:", gpus16)

    if pretrained_G14 == "":
        print("no pretrained Generator")
    if pretrained_D15 == "":
        print("no pretrained Discriminator")

    G_train = f"-pg {pretrained_G14}" if pretrained_G14 else ""
    D_train = f"-pd {pretrained_D15}" if pretrained_D15 else ""

    cmd = (
        f"{config.python_cmd} train_nsf_sim_cache_sid_load_pretrain.py -e {exp_dir1} -sr {sr2} -f0 {int(if_f0_3)} -bs {batch_size12}"
        f" -g {gpus16 if gpus16 is not None else ''} -te {total_epoch11} -se {save_epoch10} {G_train} {D_train} -l {int(if_save_latest13)}"
        f" -c {int(if_cache_gpu17)} -sw {int(if_save_every_weights18)} -v {version19} -li {log_interval}"
    )

    print(cmd)

    global p
    p = Popen(cmd, shell=True, cwd=now_dir)
    global PID
    PID = p.pid

    p.wait()

    return (
        "Training is done, check train.log",
        {"visible": False, "__type__": "update"},
        {"visible": True, "__type__": "update"},
    )


def train_index(exp_dir1, version19):
    exp_dir = os.path.join(now_dir, "logs", exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)

    feature_dim = "256" if version19 == "v1" else "768"
    feature_dir = os.path.join(exp_dir, f"3_feature{feature_dim}")

    if not os.path.exists(feature_dir) or len(os.listdir(feature_dir)) == 0:
        return "请先进行特征提取!"

    npys = [
        np.load(os.path.join(feature_dir, name))
        for name in sorted(os.listdir(feature_dir))
    ]

    big_npy = np.concatenate(npys, 0)
    np.random.shuffle(big_npy)

    infos = []
    if big_npy.shape[0] > 2 * 10**5:
        infos.append("Trying doing kmeans %s shape to 10k centers." % big_npy.shape[0])
        yield "\n".join(infos)
        try:
            big_npy = (
                MiniBatchKMeans(
                    n_clusters=10000,
                    verbose=True,
                    batch_size=256 * config.n_cpu,
                    compute_labels=False,
                    init="random",
                )
                .fit(big_npy)
                .cluster_centers_
            )
        except Exception as e:
            infos.append(str(e))
            yield "\n".join(infos)

    np.save(os.path.join(exp_dir, "total_fea.npy"), big_npy)

    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    infos.append("%s,%s" % (big_npy.shape, n_ivf))
    yield "\n".join(infos)

    index = faiss.index_factory(int(feature_dim), f"IVF{n_ivf},Flat")

    index_ivf = faiss.extract_index_ivf(index)
    index_ivf.nprobe = 1

    index.train(big_npy)

    index_file_base = f"{exp_dir}/trained_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_dir1}_{version19}.index"
    faiss.write_index(index, index_file_base)

    infos.append("adding")
    yield "\n".join(infos)

    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i : i + batch_size_add])

    index_file_base = f"{exp_dir}/added_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_dir1}_{version19}.index"
    faiss.write_index(index, index_file_base)

    infos.append(
        f"Successful Index Construction，added_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_dir1}_{version19}.index"
    )
    yield "\n".join(infos)


# def setBoolean(status): #true to false and vice versa / not implemented yet, dont touch!!!!!!!
#    status = not status
#    return status


def change_info_(ckpt_path):
    train_log_path = os.path.join(os.path.dirname(ckpt_path), "train.log")

    if not os.path.exists(train_log_path):
        return {"__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}

    try:
        with open(train_log_path, "r") as f:
            info_line = next(f).strip()
            info = eval(info_line.split("\t")[-1])

            sr, f0 = info.get("sample_rate"), info.get("if_f0")
            version = "v2" if info.get("version") == "v2" else "v1"

            return sr, str(f0), version

    except Exception as e:
        print(f"Exception occurred: {str(e)}, Traceback: {traceback.format_exc()}")
        return {"__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}


def export_onnx(model_path, exported_path):
    device = torch.device("cpu")
    checkpoint = torch.load(model_path, map_location=device)
    vec_channels = 256 if checkpoint.get("version", "v1") == "v1" else 768

    test_inputs = {
        "phone": torch.rand(1, 200, vec_channels),
        "phone_lengths": torch.LongTensor([200]),
        "pitch": torch.randint(5, 255, (1, 200)),
        "pitchf": torch.rand(1, 200),
        "ds": torch.zeros(1).long(),
        "rnd": torch.rand(1, 192, 200),
    }

    checkpoint["config"][-3] = checkpoint["weight"]["emb_g.weight"].shape[0]
    net_g = SynthesizerTrnMsNSFsidM(
        *checkpoint["config"], is_half=False, version=checkpoint.get("version", "v1")
    )

    net_g.load_state_dict(checkpoint["weight"], strict=False)
    net_g = net_g.to(device)

    dynamic_axes = {"phone": [1], "pitch": [1], "pitchf": [1], "rnd": [2]}

    torch.onnx.export(
        net_g,
        tuple(value.to(device) for value in test_inputs.values()),
        exported_path,
        dynamic_axes=dynamic_axes,
        do_constant_folding=False,
        opset_version=13,
        verbose=False,
        input_names=list(test_inputs.keys()),
        output_names=["audio"],
    )
    return "Finished"


# region Mangio-RVC-Fork CLI App

import scipy.io.wavfile as wavfile

cli_current_page = "HOME"


def cli_split_command(com):
    exp = r'(?:(?<=\s)|^)"(.*?)"(?=\s|$)|(\S+)'
    split_array = re.findall(exp, com)
    split_array = [group[0] if group[0] else group[1] for group in split_array]
    return split_array


execute_generator_function = lambda genObject: all(x is not None for x in genObject)


def cli_infer(com):
    (
        model_name,
        source_audio_path,
        output_file_name,
        feature_index_path,
        speaker_id,
        transposition,
        f0_method,
        crepe_hop_length,
        harvest_median_filter,
        resample,
        mix,
        feature_ratio,
        protection_amnt,
        _,
        do_formant,
    ) = cli_split_command(com)[:15]

    speaker_id, crepe_hop_length, harvest_median_filter, resample = map(
        int, [speaker_id, crepe_hop_length, harvest_median_filter, resample]
    )
    transposition, mix, feature_ratio, protection_amnt = map(
        float, [transposition, mix, feature_ratio, protection_amnt]
    )

    if do_formant.lower() == "false":
        Quefrency = 1.0
        Timbre = 1.0
    else:
        Quefrency, Timbre = map(float, cli_split_command(com)[15:17])

    CSVutil(
        "csvdb/formanting.csv",
        "w+",
        "formanting",
        do_formant.lower() == "true",
        Quefrency,
        Timbre,
    )

    output_message = "Mangio-RVC-Fork Infer-CLI:"
    output_path = f"audio-outputs/{output_file_name}"

    print(f"{output_message} Starting the inference...")
    vc_data = get_vc(model_name, protection_amnt, protection_amnt)
    print(vc_data)

    print(f"{output_message} Performing inference...")
    conversion_data = vc_single(
        speaker_id,
        source_audio_path,
        source_audio_path,
        transposition,
        None,  # f0 file support not implemented
        f0_method,
        feature_index_path,
        feature_index_path,
        feature_ratio,
        harvest_median_filter,
        resample,
        mix,
        protection_amnt,
        crepe_hop_length,
    )

    if "Success." in conversion_data[0]:
        print(f"{output_message} Inference succeeded. Writing to {output_path}...")
        wavfile.write(output_path, conversion_data[1][0], conversion_data[1][1])
        print(f"{output_message} Finished! Saved output to {output_path}")
    else:
        print(
            f"{output_message} Inference failed. Here's the traceback: {conversion_data[0]}"
        )


def cli_pre_process(com):
    print("Mangio-RVC-Fork Pre-process: Starting...")
    execute_generator_function(
        preprocess_dataset(*cli_split_command(com)[:3], int(cli_split_command(com)[3]))
    )
    print("Mangio-RVC-Fork Pre-process: Finished")


def cli_extract_feature(com):
    (
        model_name,
        gpus,
        num_processes,
        has_pitch_guidance,
        f0_method,
        crepe_hop_length,
        version,
    ) = cli_split_command(com)

    num_processes = int(num_processes)
    has_pitch_guidance = bool(int(has_pitch_guidance))
    crepe_hop_length = int(crepe_hop_length)

    print(
        f"Mangio-RVC-CLI: Extract Feature Has Pitch: {has_pitch_guidance}"
        f"Mangio-RVC-CLI: Extract Feature Version: {version}"
        "Mangio-RVC-Fork Feature Extraction: Starting..."
    )
    generator = extract_f0_feature(
        gpus,
        num_processes,
        f0_method,
        has_pitch_guidance,
        model_name,
        version,
        crepe_hop_length,
    )
    execute_generator_function(generator)
    print("Mangio-RVC-Fork Feature Extraction: Finished")


def cli_train(com):
    com = cli_split_command(com)
    model_name = com[0]
    sample_rate = com[1]
    bool_flags = [bool(int(i)) for i in com[2:11]]
    version = com[11]

    pretrained_base = "pretrained/" if version == "v1" else "pretrained_v2/"

    g_pretrained_path = f"{pretrained_base}f0G{sample_rate}.pth"
    d_pretrained_path = f"{pretrained_base}f0D{sample_rate}.pth"

    print("Mangio-RVC-Fork Train-CLI: Training...")
    click_train(
        model_name,
        sample_rate,
        *bool_flags,
        g_pretrained_path,
        d_pretrained_path,
        version,
    )


def cli_train_feature(com):
    output_message = "Mangio-RVC-Fork Train Feature Index-CLI"
    print(f"{output_message}: Training... Please wait")
    execute_generator_function(train_index(*cli_split_command(com)))
    print(f"{output_message}: Done!")


def cli_extract_model(com):
    extract_small_model_process = extract_small_model(*cli_split_command(com))
    print(
        "Mangio-RVC-Fork Extract Small Model: Success!"
        if extract_small_model_process == "Success."
        else f"{extract_small_model_process}\nMangio-RVC-Fork Extract Small Model: Failed!"
    )


def preset_apply(preset, qfer, tmbr):
    if preset:
        try:
            with open(preset, "r") as p:
                content = p.read().splitlines()
            qfer, tmbr = content[0], content[1]
            formant_apply(qfer, tmbr)
        except IndexError:
            print("Error: File does not have enough lines to read 'qfer' and 'tmbr'")
        except FileNotFoundError:
            print("Error: File does not exist")
        except Exception as e:
            print("An unexpected error occurred", e)

    return (
        {"value": qfer, "__type__": "update"},
        {"value": tmbr, "__type__": "update"},
    )


@jit(nopython=True)
def print_page_details():
    page_description = {
        "HOME": "\n    go home            : Takes you back to home with a navigation list."
        "\n    go infer           : Takes you to inference command execution."
        "\n    go pre-process     : Takes you to training step.1) pre-process command execution."
        "\n    go extract-feature : Takes you to training step.2) extract-feature command execution."
        "\n    go train           : Takes you to training step.3) being or continue training command execution."
        "\n    go train-feature   : Takes you to the train feature index command execution."
        "\n    go extract-model   : Takes you to the extract small model command execution.",
        "INFER": "\n    arg 1) model name with .pth in ./weights: mi-test.pth"
        "\n    arg 2) source audio path: myFolder\\MySource.wav"
        "\n    arg 3) output file name to be placed in './audio-outputs': MyTest.wav"
        "\n    arg 4) feature index file path: logs/mi-test/added_IVF3042_Flat_nprobe_1.index"
        "\n    arg 5) speaker id: 0"
        "\n    arg 6) transposition: 0"
        "\n    arg 7) f0 method: harvest (pm, harvest, crepe, crepe-tiny, hybrid[x,x,x,x], mangio-crepe, mangio-crepe-tiny, rmvpe)"
        "\n    arg 8) crepe hop length: 160"
        "\n    arg 9) harvest median filter radius: 3 (0-7)"
        "\n    arg 10) post resample rate: 0"
        "\n    arg 11) mix volume envelope: 1"
        "\n    arg 12) feature index ratio: 0.78 (0-1)"
        "\n    arg 13) Voiceless Consonant Protection (Less Artifact): 0.33 (Smaller number = more protection. 0.50 means Dont Use.)"
        "\n    arg 14) Whether to formant shift the inference audio before conversion: False (if set to false, you can ignore setting the quefrency and timbre values for formanting)"
        "\n    arg 15)* Quefrency for formanting: 8.0 (no need to set if arg14 is False/false)"
        "\n    arg 16)* Timbre for formanting: 1.2 (no need to set if arg14 is False/false) \n"
        "\nExample: mi-test.pth saudio/Sidney.wav myTest.wav logs/mi-test/added_index.index 0 -2 harvest 160 3 0 1 0.95 0.33 0.45 True 8.0 1.2",
        "PRE-PROCESS": "\n    arg 1) Model folder name in ./logs: mi-test"
        "\n    arg 2) Trainset directory: mydataset (or) E:\\my-data-set"
        "\n    arg 3) Sample rate: 40k (32k, 40k, 48k)"
        "\n    arg 4) Number of CPU threads to use: 8 \n"
        "\nExample: mi-test mydataset 40k 24",
        "EXTRACT-FEATURE": "\n    arg 1) Model folder name in ./logs: mi-test"
        "\n    arg 2) Gpu card slot: 0 (0-1-2 if using 3 GPUs)"
        "\n    arg 3) Number of CPU threads to use: 8"
        "\n    arg 4) Has Pitch Guidance?: 1 (0 for no, 1 for yes)"
        "\n    arg 5) f0 Method: harvest (pm, harvest, dio, crepe)"
        "\n    arg 6) Crepe hop length: 128"
        "\n    arg 7) Version for pre-trained models: v2 (use either v1 or v2)\n"
        "\nExample: mi-test 0 24 1 harvest 128 v2",
        "TRAIN": "\n    arg 1) Model folder name in ./logs: mi-test"
        "\n    arg 2) Sample rate: 40k (32k, 40k, 48k)"
        "\n    arg 3) Has Pitch Guidance?: 1 (0 for no, 1 for yes)"
        "\n    arg 4) speaker id: 0"
        "\n    arg 5) Save epoch iteration: 50"
        "\n    arg 6) Total epochs: 10000"
        "\n    arg 7) Batch size: 8"
        "\n    arg 8) Gpu card slot: 0 (0-1-2 if using 3 GPUs)"
        "\n    arg 9) Save only the latest checkpoint: 0 (0 for no, 1 for yes)"
        "\n    arg 10) Whether to cache training set to vram: 0 (0 for no, 1 for yes)"
        "\n    arg 11) Save extracted small model every generation?: 0 (0 for no, 1 for yes)"
        "\n    arg 12) Model architecture version: v2 (use either v1 or v2)\n"
        "\nExample: mi-test 40k 1 0 50 10000 8 0 0 0 0 v2",
        "TRAIN-FEATURE": "\n    arg 1) Model folder name in ./logs: mi-test"
        "\n    arg 2) Model architecture version: v2 (use either v1 or v2)\n"
        "\nExample: mi-test v2",
        "EXTRACT-MODEL": "\n    arg 1) Model Path: logs/mi-test/G_168000.pth"
        "\n    arg 2) Model save name: MyModel"
        "\n    arg 3) Sample rate: 40k (32k, 40k, 48k)"
        "\n    arg 4) Has Pitch Guidance?: 1 (0 for no, 1 for yes)"
        '\n    arg 5) Model information: "My Model"'
        "\n    arg 6) Model architecture version: v2 (use either v1 or v2)\n"
        '\nExample: logs/mi-test/G_168000.pth MyModel 40k 1 "Created by Cole Mangio" v2',
    }
    print(page_description.get(cli_current_page, "Invalid page"))


def change_page(page):
    global cli_current_page
    cli_current_page = page
    return 0


@jit
def execute_command(com):
    command_to_page = {
        "go home": "HOME",
        "go infer": "INFER",
        "go pre-process": "PRE-PROCESS",
        "go extract-feature": "EXTRACT-FEATURE",
        "go train": "TRAIN",
        "go train-feature": "TRAIN-FEATURE",
        "go extract-model": "EXTRACT-MODEL",
    }

    page_to_function = {
        "INFER": cli_infer,
        "PRE-PROCESS": cli_pre_process,
        "EXTRACT-FEATURE": cli_extract_feature,
        "TRAIN": cli_train,
        "TRAIN-FEATURE": cli_train_feature,
        "EXTRACT-MODEL": cli_extract_model,
    }

    if com in command_to_page:
        return change_page(command_to_page[com])

    if com[:3] == "go ":
        print(f"page '{com[3:]}' does not exist!")
        return 0

    if cli_current_page in page_to_function:
        page_to_function[cli_current_page](com)


def cli_navigation_loop():
    while True:
        print(f"\nYou are currently in '{cli_current_page}':")
        print_page_details()
        print(f"{cli_current_page}: ", end="")
        try:
            execute_command(input())
        except Exception as e:
            print(f"An error occurred: {traceback.format_exc()}")


if config.is_cli:
    print(
        "\n\nMangio-RVC-Fork v2 CLI App!\n"
        "Welcome to the CLI version of RVC. Please read the documentation on https://github.com/Mangio621/Mangio-RVC-Fork (README.MD) to understand how to use this app.\n"
    )
    cli_navigation_loop()

# endregion

# region RVC WebUI App
"""
def get_presets():
    data = None
    with open('../inference-presets.json', 'r') as file:
        data = json.load(file)
    preset_names = []
    for preset in data['presets']:
        preset_names.append(preset['name'])
    
    return preset_names
"""


def match_index(sid0):
    folder = sid0.split(".")[0].split("_")[0]
    parent_dir = "./logs/" + folder
    if not os.path.exists(parent_dir):
        return "", ""

    for filename in os.listdir(parent_dir):
        if filename.endswith(".index"):
            index_path = os.path.join(parent_dir, filename).replace("\\", "/")
            print(index_path)
            if index_path in indexes_list:
                return index_path, index_path

            lowered_index_path = os.path.join(parent_dir.lower(), filename).replace(
                "\\", "/"
            )
            if lowered_index_path in indexes_list:
                return lowered_index_path, lowered_index_path
    return "", ""


def stoptraining(mim):
    if mim:
        try:
            CSVutil("csvdb/stop.csv", "w+", "stop", "True")
            os.kill(PID, SIGTERM)
        except Exception as e:
            print(f"Couldn't click due to {e}")
    return (
        {"visible": False, "__type__": "update"},
        {"visible": True, "__type__": "update"},
    )


tab_faq = i18n("常见问题解答")
faq_file = "docs/faq.md" if tab_faq == "常见问题解答" else "docs/faq_en.md"
weights_dir = "weights/"
