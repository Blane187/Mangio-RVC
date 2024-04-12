from original import *



def GradioSetup():
    # Change your Gradio Theme here. 👇 👇 👇 👇 Example: " theme='HaleyCH/HaleyCH_Theme' "
    with gr.Blocks(theme=gr.themes.Soft(), title="Mangio-RVC-Web 💻") as app:
        gr.HTML("<h1> The Mangio-RVC-Fork 💻 </h1>")
        gr.Markdown(
            value=i18n(
                "本软件以MIT协议开源, 作者不对软件具备任何控制力, 使用软件者、传播软件导出的声音者自负全责. <br>如不认可该条款, 则不能使用或引用软件包内任何代码和文件. 详见根目录<b>使用需遵守的协议-LICENSE.txt</b>."
            )
        )
        with gr.Tabs():
            with gr.TabItem(i18n("模型推理")):
                # Inference Preset Row
                # with gr.Row():
                #     mangio_preset = gr.Dropdown(label="Inference Preset", choices=sorted(get_presets()))
                #     mangio_preset_name_save = gr.Textbox(
                #         label="Your preset name"
                #     )
                #     mangio_preset_save_btn = gr.Button('Save Preset', variant="primary")

                # Other RVC stuff
                with gr.Row():
                    sid0 = gr.Dropdown(
                        label=i18n("推理音色"), choices=sorted(names), value=""
                    )

                    refresh_button = gr.Button(
                        i18n("Refresh voice list, index path and audio files"),
                        variant="primary",
                    )
                    clean_button = gr.Button(i18n("卸载音色省显存"), variant="primary")
                    spk_item = gr.Slider(
                        minimum=0,
                        maximum=2333,
                        step=1,
                        label=i18n("请选择说话人id"),
                        value=0,
                        visible=False,
                        interactive=True,
                    )
                    clean_button.click(
                        fn=lambda: ({"value": "", "__type__": "update"}),
                        inputs=[],
                        outputs=[sid0],
                    )

                with gr.Group():
                    gr.Markdown(
                        value=i18n(
                            "男转女推荐+12key, 女转男推荐-12key, 如果音域爆炸导致音色失真也可以自己调整到合适音域. "
                        )
                    )
                    with gr.Row():
                        with gr.Column():
                            vc_transform0 = gr.Number(
                                label=i18n("变调(整数, 半音数量, 升八度12降八度-12)"), value=0
                            )
                            input_audio0 = gr.Textbox(
                                label=i18n(
                                    "Add audio's name to the path to the audio file to be processed (default is the correct format example) Remove the path to use an audio from the dropdown list:"
                                ),
                                value=os.path.abspath(os.getcwd()).replace("\\", "/")
                                + "/audios/"
                                + "audio.wav",
                            )
                            input_audio1 = gr.Dropdown(
                                label=i18n(
                                    "Auto detect audio path and select from the dropdown:"
                                ),
                                choices=sorted(audio_paths),
                                value="",
                                interactive=True,
                            )
                            input_audio1.change(
                                fn=lambda: "", inputs=[], outputs=[input_audio0]
                            )
                            f0method0 = gr.Radio(
                                label=i18n(
                                    "选择音高提取算法,输入歌声可用pm提速,harvest低音好但巨慢无比,crepe效果好但吃GPU"
                                ),
                                choices=[
                                    "pm",
                                    "harvest",
                                    "dio",
                                    "crepe",
                                    "crepe-tiny",
                                    "mangio-crepe",
                                    "mangio-crepe-tiny",
                                    "rmvpe",
                                    "hybrid[rmvpe+mangio-crepe]",
                                ],  # Fork Feature. Add Crepe-Tiny
                                value="rmvpe",
                                interactive=True,
                            )
                            crepe_hop_length = gr.Slider(
                                minimum=1,
                                maximum=512,
                                step=1,
                                label=i18n("crepe_hop_length"),
                                value=120,
                                interactive=True,
                                visible=False,
                            )
                            f0method0.change(
                                fn=lambda radio: (
                                    {
                                        "visible": radio
                                        in ["mangio-crepe", "mangio-crepe-tiny"],
                                        "__type__": "update",
                                    }
                                ),
                                inputs=[f0method0],
                                outputs=[crepe_hop_length],
                            )
                            filter_radius0 = gr.Slider(
                                minimum=0,
                                maximum=7,
                                label=i18n(
                                    ">=3则使用对harvest音高识别的结果使用中值滤波，数值为滤波半径，使用可以削弱哑音"
                                ),
                                value=3,
                                step=1,
                                interactive=True,
                            )
                        with gr.Column():
                            file_index1 = gr.Textbox(
                                label=i18n("特征检索库文件路径,为空则使用下拉的选择结果"),
                                value="",
                                interactive=True,
                            )

                            file_index2 = gr.Dropdown(
                                label="3. Path to your added.index file (if it didn't automatically find it.)",
                                choices=get_indexes(),
                                interactive=True,
                                allow_custom_value=True,
                            )
                            # sid0.select(fn=match_index, inputs=sid0, outputs=file_index2)

                            refresh_button.click(
                                fn=change_choices,
                                inputs=[],
                                outputs=[sid0, file_index2, input_audio1],
                            )
                            # file_big_npy1 = gr.Textbox(
                            #     label=i18n("特征文件路径"),
                            #     value="E:\\codes\py39\\vits_vc_gpu_train\\logs\\mi-test-1key\\total_fea.npy",
                            #     interactive=True,
                            # )
                            index_rate1 = gr.Slider(
                                minimum=0,
                                maximum=1,
                                label=i18n("检索特征占比"),
                                value=0.75,
                                interactive=True,
                            )
                        with gr.Column():
                            resample_sr0 = gr.Slider(
                                minimum=0,
                                maximum=48000,
                                label=i18n("后处理重采样至最终采样率，0为不进行重采样"),
                                value=0,
                                step=1,
                                interactive=True,
                            )
                            rms_mix_rate0 = gr.Slider(
                                minimum=0,
                                maximum=1,
                                label=i18n("输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络"),
                                value=0.25,
                                interactive=True,
                            )
                            protect0 = gr.Slider(
                                minimum=0,
                                maximum=0.5,
                                label=i18n(
                                    "保护清辅音和呼吸声，防止电音撕裂等artifact，拉满0.5不开启，调低加大保护力度但可能降低索引效果"
                                ),
                                value=0.33,
                                step=0.01,
                                interactive=True,
                            )
                            formanting = gr.Checkbox(
                                value=bool(DoFormant),
                                label="[EXPERIMENTAL] Formant shift inference audio",
                                info="Used for male to female and vice-versa conversions",
                                interactive=True,
                                visible=True,
                            )

                            formant_preset = gr.Dropdown(
                                value="",
                                choices=get_fshift_presets(),
                                label="Browse presets for formanting",
                                info="Presets are located in formantshiftcfg/ folder",
                                visible=bool(DoFormant),
                            )

                            formant_refresh_button = gr.Button(
                                value="\U0001f504",
                                visible=bool(DoFormant),
                                variant="primary",
                            )

                            qfrency = gr.Slider(
                                value=Quefrency,
                                info="Default value is 1.0",
                                label="Quefrency for formant shifting",
                                minimum=0.0,
                                maximum=16.0,
                                step=0.1,
                                visible=bool(DoFormant),
                                interactive=True,
                            )

                            tmbre = gr.Slider(
                                value=Timbre,
                                info="Default value is 1.0",
                                label="Timbre for formant shifting",
                                minimum=0.0,
                                maximum=16.0,
                                step=0.1,
                                visible=bool(DoFormant),
                                interactive=True,
                            )

                            formant_preset.change(
                                fn=preset_apply,
                                inputs=[formant_preset, qfrency, tmbre],
                                outputs=[qfrency, tmbre],
                            )
                            frmntbut = gr.Button(
                                "Apply", variant="primary", visible=bool(DoFormant)
                            )
                            formanting.change(
                                fn=formant_enabled,
                                inputs=[
                                    formanting,
                                    qfrency,
                                    tmbre,
                                    frmntbut,
                                    formant_preset,
                                    formant_refresh_button,
                                ],
                                outputs=[
                                    formanting,
                                    qfrency,
                                    tmbre,
                                    frmntbut,
                                    formant_preset,
                                    formant_refresh_button,
                                ],
                            )
                            frmntbut.click(
                                fn=formant_apply,
                                inputs=[qfrency, tmbre],
                                outputs=[qfrency, tmbre],
                            )
                            formant_refresh_button.click(
                                fn=update_fshift_presets,
                                inputs=[formant_preset, qfrency, tmbre],
                                outputs=[formant_preset, qfrency, tmbre],
                            )

                        f0_file = gr.File(label=i18n("F0曲线文件, 可选, 一行一个音高, 代替默认F0及升降调"))
                        with gr.Row():
                        but0 = gr.Button(i18n("转换"), variant="primary")
                        with gr.Row():
                            vc_output1 = gr.Textbox(label=i18n("输出信息"))
                        with gr.Row():
                            vc_output2 = gr.Audio(label=i18n("输出音频(右下角三个点,点了可以下载)"))
                        but0.click(
                            vc_single,
                            [
                                spk_item,
                                input_audio0,
                                input_audio1,
                                vc_transform0,
                                f0_file,
                                f0method0,
                                file_index1,
                                file_index2,
                                # file_big_npy1,
                                index_rate1,
                                filter_radius0,
                                resample_sr0,
                                rms_mix_rate0,
                                protect0,
                                crepe_hop_length,
                            ],
                            [vc_output1, vc_output2],
                        )
                with gr.Group():
                    gr.Markdown(
                        value=i18n(
                            "批量转换, 输入待转换音频文件夹, 或上传多个音频文件, 在指定文件夹(默认opt)下输出转换的音频. "
                        )
                    )
                    with gr.Row():
                        with gr.Column():
                            vc_transform1 = gr.Number(
                                label=i18n("变调(整数, 半音数量, 升八度12降八度-12)"), value=0
                            )
                            opt_input = gr.Textbox(label=i18n("指定输出文件夹"), value="opt")
                            f0method1 = gr.Radio(
                                label=i18n(
                                    "选择音高提取算法,输入歌声可用pm提速,harvest低音好但巨慢无比,crepe效果好但吃GPU"
                                ),
                                choices=["pm", "harvest", "crepe", "rmvpe"],
                                value="rmvpe",
                                interactive=True,
                            )

                            filter_radius1 = gr.Slider(
                                minimum=0,
                                maximum=7,
                                label=i18n(
                                    ">=3则使用对harvest音高识别的结果使用中值滤波，数值为滤波半径，使用可以削弱哑音"
                                ),
                                value=3,
                                step=1,
                                interactive=True,
                            )
                        with gr.Column():
                            file_index3 = gr.Textbox(
                                label=i18n("特征检索库文件路径,为空则使用下拉的选择结果"),
                                value="",
                                interactive=True,
                            )
                            file_index4 = gr.Dropdown(  # file index dropdown for batch
                                label=i18n("自动检测index路径,下拉式选择(dropdown)"),
                                choices=get_indexes(),
                                interactive=True,
                            )
                            sid0.select(
                                fn=match_index,
                                inputs=[sid0],
                                outputs=[file_index2, file_index4],
                            )
                            refresh_button.click(
                                fn=lambda: change_choices()[1],
                                inputs=[],
                                outputs=file_index4,
                            )
                            # file_big_npy2 = gr.Textbox(
                            #     label=i18n("特征文件路径"),
                            #     value="E:\\codes\\py39\\vits_vc_gpu_train\\logs\\mi-test-1key\\total_fea.npy",
                            #     interactive=True,
                            # )
                            index_rate2 = gr.Slider(
                                minimum=0,
                                maximum=1,
                                label=i18n("检索特征占比"),
                                value=1,
                                interactive=True,
                            )
                        with gr.Column():
                            resample_sr1 = gr.Slider(
                                minimum=0,
                                maximum=48000,
                                label=i18n("后处理重采样至最终采样率，0为不进行重采样"),
                                value=0,
                                step=1,
                                interactive=True,
                            )
                            rms_mix_rate1 = gr.Slider(
                                minimum=0,
                                maximum=1,
                                label=i18n("输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络"),
                                value=1,
                                interactive=True,
                            )
                            protect1 = gr.Slider(
                                minimum=0,
                                maximum=0.5,
                                label=i18n(
                                    "保护清辅音和呼吸声，防止电音撕裂等artifact，拉满0.5不开启，调低加大保护力度但可能降低索引效果"
                                ),
                                value=0.33,
                                step=0.01,
                                interactive=True,
                            )
                        with gr.Column():
                            dir_input = gr.Textbox(
                                label=i18n("输入待处理音频文件夹路径(去文件管理器地址栏拷就行了)"),
                                value=os.path.abspath(os.getcwd()).replace("\\", "/")
                                + "/audios/",
                            )
                            inputs = gr.File(
                                file_count="multiple",
                                label=i18n("也可批量输入音频文件, 二选一, 优先读文件夹"),
                            )
                        with gr.Row():
                            format1 = gr.Radio(
                                label=i18n("导出文件格式"),
                                choices=["wav", "flac", "mp3", "m4a"],
                                value="flac",
                                interactive=True,
                            )
                            but1 = gr.Button(i18n("转换"), variant="primary")
                            vc_output3 = gr.Textbox(label=i18n("输出信息"))
                        but1.click(
                            vc_multi,
                            [
                                spk_item,
                                dir_input,
                                opt_input,
                                inputs,
                                vc_transform1,
                                f0method1,
                                file_index3,
                                file_index4,
                                # file_big_npy2,
                                index_rate2,
                                filter_radius1,
                                resample_sr1,
                                rms_mix_rate1,
                                protect1,
                                format1,
                                crepe_hop_length,
                            ],
                            [vc_output3],
                        )
                sid0.change(
                    fn=get_vc,
                    inputs=[sid0, protect0, protect1],
                    outputs=[spk_item, protect0, protect1],
                )
            with gr.TabItem(i18n("伴奏人声分离&去混响&去回声")):
                with gr.Group():
                    gr.Markdown(
                        value=i18n(
                            "人声伴奏分离批量处理， 使用UVR5模型。 <br>"
                            "合格的文件夹路径格式举例： E:\\codes\\py39\\vits_vc_gpu\\白鹭霜华测试样例(去文件管理器地址栏拷就行了)。 <br>"
                            "模型分为三类： <br>"
                            "1、保留人声：不带和声的音频选这个，对主人声保留比HP5更好。内置HP2和HP3两个模型，HP3可能轻微漏伴奏但对主人声保留比HP2稍微好一丁点； <br>"
                            "2、仅保留主人声：带和声的音频选这个，对主人声可能有削弱。内置HP5一个模型； <br> "
                            "3、去混响、去延迟模型（by FoxJoy）：<br>"
                            "  (1)MDX-Net(onnx_dereverb):对于双通道混响是最好的选择，不能去除单通道混响；<br>"
                            "&emsp;(234)DeEcho:去除延迟效果。Aggressive比Normal去除得更彻底，DeReverb额外去除混响，可去除单声道混响，但是对高频重的板式混响去不干净。<br>"
                            "去混响/去延迟，附：<br>"
                            "1、DeEcho-DeReverb模型的耗时是另外2个DeEcho模型的接近2倍；<br>"
                            "2、MDX-Net-Dereverb模型挺慢的；<br>"
                            "3、个人推荐的最干净的配置是先MDX-Net再DeEcho-Aggressive。"
                        )
                    )
                    with gr.Row():
                        with gr.Column():
                            dir_wav_input = gr.Textbox(
                                label=i18n("输入待处理音频文件夹路径"),
                                value=((os.getcwd()).replace("\\", "/") + "/audios/"),
                            )
                            wav_inputs = gr.File(
                                file_count="multiple",
                                label=i18n("也可批量输入音频文件, 二选一, 优先读文件夹"),
                            )  #####
                        with gr.Column():
                            model_choose = gr.Dropdown(
                                label=i18n("模型"), choices=uvr5_names
                            )
                            agg = gr.Slider(
                                minimum=0,
                                maximum=20,
                                step=1,
                                label="人声提取激进程度",
                                value=10,
                                interactive=True,
                                visible=False,  # 先不开放调整
                            )
                            opt_vocal_root = gr.Textbox(
                                label=i18n("指定输出主人声文件夹"), value="opt"
                            )
                            opt_ins_root = gr.Textbox(
                                label=i18n("指定输出非主人声文件夹"), value="opt"
                            )
                            format0 = gr.Radio(
                                label=i18n("导出文件格式"),
                                choices=["wav", "flac", "mp3", "m4a"],
                                value="flac",
                                interactive=True,
                            )
                        but2 = gr.Button(i18n("转换"), variant="primary")
                        vc_output4 = gr.Textbox(label=i18n("输出信息"))
                        but2.click(
                            uvr,
                            [
                                model_choose,
                                dir_wav_input,
                                opt_vocal_root,
                                wav_inputs,
                                opt_ins_root,
                                agg,
                                format0,
                            ],
                            [vc_output4],
                        )
            with gr.TabItem(i18n("训练")):
                gr.Markdown(
                    value=i18n(
                        "step1: 填写实验配置. 实验数据放在logs下, 每个实验一个文件夹, 需手工输入实验名路径, 内含实验配置, 日志, 训练得到的模型文件. "
                    )
                )
                with gr.Row():
                    exp_dir1 = gr.Textbox(label=i18n("输入实验名"), value="mi-test")
                    sr2 = gr.Radio(
                        label=i18n("目标采样率"),
                        choices=["40k", "48k", "32k"],
                        value="40k",
                        interactive=True,
                    )
                    if_f0_3 = gr.Checkbox(
                        label="Whether the model has pitch guidance.",
                        value=True,
                        interactive=True,
                    )
                    version19 = gr.Radio(
                        label=i18n("版本"),
                        choices=["v1", "v2"],
                        value="v2",
                        interactive=True,
                        visible=True,
                    )
                    np7 = gr.Slider(
                        minimum=0,
                        maximum=config.n_cpu,
                        step=1,
                        label=i18n("提取音高和处理数据使用的CPU进程数"),
                        value=int(np.ceil(config.n_cpu / 1.5)),
                        interactive=True,
                    )
                with gr.Group():  # 暂时单人的, 后面支持最多4人的#数据处理
                    gr.Markdown(
                        value=i18n(
                            "step2a: 自动遍历训练文件夹下所有可解码成音频的文件并进行切片归一化, 在实验目录下生成2个wav文件夹; 暂时只支持单人训练. "
                        )
                    )
                    with gr.Row():
                        trainset_dir4 = gr.Textbox(
                            label=i18n("输入训练文件夹路径"),
                            value=os.path.abspath(os.getcwd()) + "\\datasets\\",
                        )
                        spk_id5 = gr.Slider(
                            minimum=0,
                            maximum=4,
                            step=1,
                            label=i18n("请指定说话人id"),
                            value=0,
                            interactive=True,
                        )
                        but1 = gr.Button(i18n("处理数据"), variant="primary")
                        info1 = gr.Textbox(label=i18n("输出信息"), value="")
                        but1.click(
                            preprocess_dataset,
                            [trainset_dir4, exp_dir1, sr2, np7],
                            [info1],
                        )
                with gr.Group():
                    step2b = gr.Markdown(
                        value=i18n("step2b: 使用CPU提取音高(如果模型带音高), 使用GPU提取特征(选择卡号)")
                    )
                    with gr.Row():
                        with gr.Column():
                            gpus6 = gr.Textbox(
                                label=i18n("以-分隔输入使用的卡号, 例如   0-1-2   使用卡0和卡1和卡2"),
                                value=gpus,
                                interactive=True,
                            )
                            gpu_info9 = gr.Textbox(label=i18n("显卡信息"), value=gpu_info)
                        with gr.Column():
                            f0method8 = gr.Radio(
                                label=i18n(
                                    "选择音高提取算法:输入歌声可用pm提速,高质量语音但CPU差可用dio提速,harvest质量更好但慢"
                                ),
                                choices=[
                                    "pm",
                                    "harvest",
                                    "dio",
                                    "crepe",
                                    "mangio-crepe",
                                    "rmvpe",
                                ],  # Fork feature: Crepe on f0 extraction for training.
                                value="rmvpe",
                                interactive=True,
                            )

                            extraction_crepe_hop_length = gr.Slider(
                                minimum=1,
                                maximum=512,
                                step=1,
                                label=i18n("crepe_hop_length"),
                                value=64,
                                interactive=True,
                                visible=False,
                            )

                            f0method8.change(
                                fn=lambda radio: (
                                    {
                                        "visible": radio
                                        in ["mangio-crepe", "mangio-crepe-tiny"],
                                        "__type__": "update",
                                    }
                                ),
                                inputs=[f0method8],
                                outputs=[extraction_crepe_hop_length],
                            )
                        but2 = gr.Button(i18n("特征提取"), variant="primary")
                        info2 = gr.Textbox(
                            label=i18n("输出信息"), value="", max_lines=8, interactive=False
                        )
                        but2.click(
                            extract_f0_feature,
                            [
                                gpus6,
                                np7,
                                f0method8,
                                if_f0_3,
                                exp_dir1,
                                version19,
                                extraction_crepe_hop_length,
                            ],
                            [info2],
                        )
                with gr.Group():
                    gr.Markdown(value=i18n("step3: 填写训练设置, 开始训练模型和索引"))
                    with gr.Row():
                        save_epoch10 = gr.Slider(
                            minimum=1,
                            maximum=50,
                            step=1,
                            label=i18n("保存频率save_every_epoch"),
                            value=5,
                            interactive=True,
                            visible=True,
                        )
                        total_epoch11 = gr.Slider(
                            minimum=1,
                            maximum=10000,
                            step=1,
                            label=i18n("总训练轮数total_epoch"),
                            value=20,
                            interactive=True,
                        )
                        batch_size12 = gr.Slider(
                            minimum=1,
                            maximum=40,
                            step=1,
                            label=i18n("每张显卡的batch_size"),
                            value=default_batch_size,
                            interactive=True,
                        )
                        if_save_latest13 = gr.Checkbox(
                            label="Whether to save only the latest .ckpt file to save hard drive space",
                            value=True,
                            interactive=True,
                        )
                        if_cache_gpu17 = gr.Checkbox(
                            label="Cache all training sets to GPU memory. Caching small datasets (less than 10 minutes) can speed up training, but caching large datasets will consume a lot of GPU memory and may not provide much speed improvement",
                            value=False,
                            interactive=True,
                        )
                        if_save_every_weights18 = gr.Checkbox(
                            label="Save a small final model to the 'weights' folder at each save point",
                            value=True,
                            interactive=True,
                        )
                    with gr.Row():
                        pretrained_G14 = gr.Textbox(
                            lines=2,
                            label=i18n("加载预训练底模G路径"),
                            value="pretrained/f0G40k.pth",
                            interactive=True,
                        )
                        pretrained_D15 = gr.Textbox(
                            lines=2,
                            label=i18n("加载预训练底模D路径"),
                            value="pretrained/f0D40k.pth",
                            interactive=True,
                        )
                        sr2.change(
                            change_sr2,
                            [sr2, if_f0_3, version19],
                            [pretrained_G14, pretrained_D15],
                        )
                        version19.change(
                            change_version19,
                            [sr2, if_f0_3, version19],
                            [pretrained_G14, pretrained_D15, sr2],
                        )
                        if_f0_3.change(
                            fn=change_f0,
                            inputs=[if_f0_3, sr2, version19],
                            outputs=[f0method8, pretrained_G14, pretrained_D15],
                        )
                        if_f0_3.change(
                            fn=lambda radio: (
                                {
                                    "visible": radio
                                    in ["mangio-crepe", "mangio-crepe-tiny"],
                                    "__type__": "update",
                                }
                            ),
                            inputs=[f0method8],
                            outputs=[extraction_crepe_hop_length],
                        )
                        gpus16 = gr.Textbox(
                            label=i18n("以-分隔输入使用的卡号, 例如   0-1-2   使用卡0和卡1和卡2"),
                            value=gpus,
                            interactive=True,
                        )
                        butstop = gr.Button(
                            "Stop Training",
                            variant="primary",
                            visible=False,
                        )
                        but3 = gr.Button(i18n("训练模型"), variant="primary", visible=True)
                        but3.click(
                            fn=stoptraining,
                            inputs=[gr.Number(value=0, visible=False)],
                            outputs=[but3, butstop],
                        )
                        butstop.click(
                            fn=stoptraining,
                            inputs=[gr.Number(value=1, visible=False)],
                            outputs=[butstop, but3],
                        )

                        but4 = gr.Button(i18n("训练特征索引"), variant="primary")
                        # but5 = gr.Button(i18n("一键训练"), variant="primary")
                        info3 = gr.Textbox(label=i18n("输出信息"), value="", max_lines=10)

                        if_save_every_weights18.change(
                            fn=lambda if_save_every_weights: (
                                {"visible": if_save_every_weights, "__type__": "update"}
                            ),
                            inputs=[if_save_every_weights18],
                            outputs=[save_epoch10],
                        )

                        but3.click(
                            click_train,
                            [
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
                            ],
                            [info3, butstop, but3],
                        )

                        but4.click(train_index, [exp_dir1, version19], info3)

                        # but5.click(
                        #    train1key,
                        #    [
                        #        exp_dir1,
                        #        sr2,
                        #        if_f0_3,
                        #        trainset_dir4,
                        #        spk_id5,
                        #        np7,
                        #        f0method8,
                        #        save_epoch10,
                        #        total_epoch11,
                        #        batch_size12,
                        #        if_save_latest13,
                        #        pretrained_G14,
                        #        pretrained_D15,
                        #        gpus16,
                        #        if_cache_gpu17,
                        #        if_save_every_weights18,
                        #        version19,
                        #        extraction_crepe_hop_length
                        #    ],
                        #    info3,
                        # )
                with gr.Group():
                    gr.Markdown(
                        value="Step 4: Export lowest points on a graph of the model "
                        "After clicking on Export lowest points of a model, "
                        "The new files will be located in logs/[yourmodelname]/lowestvals/ folder"
                    )

                    with gr.Row():
                        with gr.Accordion(label="Lowest points export"):
                            lowestval_weight_dir = gr.Textbox(visible=False)
                            ds = gr.Textbox(visible=False)
                            weights_dir1 = gr.Textbox(visible=False, value=weights_dir)

                            with gr.Row():
                                amntlastmdls = gr.Slider(
                                    minimum=1,
                                    maximum=25,
                                    label="How many lowest points to save",
                                    value=3,
                                    step=1,
                                    interactive=True,
                                )
                                lpexport = gr.Button(
                                    value="Export lowest points of a model",
                                    variant="primary",
                                )
                                lw_mdls = gr.File(
                                    file_count="multiple",
                                    label="Output models",
                                    interactive=False,
                                )  #####

                            with gr.Row():
                                infolpex = gr.Textbox(
                                    label="Output information:", value="", max_lines=10
                                )
                                mdlbl = gr.Dataframe(
                                    label="Stats of selected models",
                                    datatype="number",
                                    type="pandas",
                                )

                            lpexport.click(
                                lambda model_name: os.path.join(
                                    "logs", model_name, "lowestvals"
                                ),
                                inputs=[exp_dir1],
                                outputs=[lowestval_weight_dir],
                            )

                            lpexport.click(
                                fn=tensorlowest.main,
                                inputs=[exp_dir1, save_epoch10, amntlastmdls],
                                outputs=[ds],
                            )

                            ds.change(
                                fn=tensorlowest.selectweights,
                                inputs=[
                                    exp_dir1,
                                    ds,
                                    weights_dir1,
                                    lowestval_weight_dir,
                                ],
                                outputs=[infolpex, lw_mdls, mdlbl],
                            )
            with gr.TabItem(i18n("ckpt处理")):
                with gr.Group():
                    gr.Markdown(value=i18n("模型融合, 可用于测试音色融合"))
                    with gr.Row():
                        ckpt_a = gr.Textbox(
                            label=i18n("A模型路径"),
                            value="",
                            interactive=True,
                            placeholder="Path to your model A.",
                        )
                        ckpt_b = gr.Textbox(
                            label=i18n("B模型路径"),
                            value="",
                            interactive=True,
                            placeholder="Path to your model B.",
                        )
                        alpha_a = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label=i18n("A模型权重"),
                            value=0.5,
                            interactive=True,
                        )
                    with gr.Row():
                        sr_ = gr.Radio(
                            label=i18n("目标采样率"),
                            choices=["40k", "48k"],
                            value="40k",
                            interactive=True,
                        )
                        if_f0_ = gr.Checkbox(
                            label="Whether the model has pitch guidance.",
                            value=True,
                            interactive=True,
                        )
                        info__ = gr.Textbox(
                            label=i18n("要置入的模型信息"),
                            value="",
                            max_lines=8,
                            interactive=True,
                            placeholder="Model information to be placed.",
                        )
                        name_to_save0 = gr.Textbox(
                            label=i18n("保存的模型名不带后缀"),
                            value="",
                            placeholder="Name for saving.",
                            max_lines=1,
                            interactive=True,
                        )
                        version_2 = gr.Radio(
                            label=i18n("模型版本型号"),
                            choices=["v1", "v2"],
                            value="v1",
                            interactive=True,
                        )
                    with gr.Row():
                        but6 = gr.Button(i18n("融合"), variant="primary")
                        info4 = gr.Textbox(label=i18n("输出信息"), value="", max_lines=8)
                    but6.click(
                        merge,
                        [
                            ckpt_a,
                            ckpt_b,
                            alpha_a,
                            sr_,
                            if_f0_,
                            info__,
                            name_to_save0,
                            version_2,
                        ],
                        info4,
                    )  # def merge(path1,path2,alpha1,sr,f0,info):
                with gr.Group():
                    gr.Markdown(value=i18n("修改模型信息(仅支持weights文件夹下提取的小模型文件)"))
                    with gr.Row():  ######
                        ckpt_path0 = gr.Textbox(
                            label=i18n("模型路径"),
                            placeholder="Path to your Model.",
                            value="",
                            interactive=True,
                        )
                        info_ = gr.Textbox(
                            label=i18n("要改的模型信息"),
                            value="",
                            max_lines=8,
                            interactive=True,
                            placeholder="Model information to be changed.",
                        )
                        name_to_save1 = gr.Textbox(
                            label=i18n("保存的文件名, 默认空为和源文件同名"),
                            placeholder="Either leave empty or put in the Name of the Model to be saved.",
                            value="",
                            max_lines=8,
                            interactive=True,
                        )
                    with gr.Row():
                        but7 = gr.Button(i18n("修改"), variant="primary")
                        info5 = gr.Textbox(label=i18n("输出信息"), value="", max_lines=8)
                    but7.click(change_info, [ckpt_path0, info_, name_to_save1], info5)
                with gr.Group():
                    gr.Markdown(value=i18n("查看模型信息(仅支持weights文件夹下提取的小模型文件)"))
                    with gr.Row():
                        ckpt_path1 = gr.Textbox(
                            label=i18n("模型路径"),
                            value="",
                            interactive=True,
                            placeholder="Model path here.",
                        )
                        but8 = gr.Button(i18n("查看"), variant="primary")
                        info6 = gr.Textbox(label=i18n("输出信息"), value="", max_lines=8)
                    but8.click(show_info, [ckpt_path1], info6)
                with gr.Group():
                    gr.Markdown(
                        value=i18n(
                            "模型提取(输入logs文件夹下大文件模型路径),适用于训一半不想训了模型没有自动提取保存小文件模型,或者想测试中间模型的情况"
                        )
                    )
                    with gr.Row():
                        ckpt_path2 = gr.Textbox(
                            lines=3,
                            label=i18n("模型路径"),
                            value=os.path.abspath(os.getcwd()).replace("\\", "/")
                            + "/logs/[YOUR_MODEL]/G_23333.pth",
                            interactive=True,
                        )
                        save_name = gr.Textbox(
                            label=i18n("保存名"),
                            value="",
                            interactive=True,
                            placeholder="Your filename here.",
                        )
                        sr__ = gr.Radio(
                            label=i18n("目标采样率"),
                            choices=["32k", "40k", "48k"],
                            value="40k",
                            interactive=True,
                        )
                        if_f0__ = gr.Checkbox(
                            label="Whether the model has pitch guidance.",
                            value=True,
                            interactive=True,
                        )
                        version_1 = gr.Radio(
                            label=i18n("模型版本型号"),
                            choices=["v1", "v2"],
                            value="v2",
                            interactive=True,
                        )
                        info___ = gr.Textbox(
                            label=i18n("要置入的模型信息"),
                            value="",
                            max_lines=8,
                            interactive=True,
                            placeholder="Model info here.",
                        )
                        but9 = gr.Button(i18n("提取"), variant="primary")
                        info7 = gr.Textbox(label=i18n("输出信息"), value="", max_lines=8)
                        ckpt_path2.change(
                            change_info_, [ckpt_path2], [sr__, if_f0__, version_1]
                        )
                    but9.click(
                        extract_small_model,
                        [ckpt_path2, save_name, sr__, if_f0__, info___, version_1],
                        info7,
                    )

            with gr.TabItem(i18n("Onnx导出")):
                with gr.Row():
                    ckpt_dir = gr.Textbox(
                        label=i18n("RVC模型路径"),
                        value="",
                        interactive=True,
                        placeholder="RVC model path.",
                    )
                with gr.Row():
                    onnx_dir = gr.Textbox(
                        label=i18n("Onnx输出路径"),
                        value="",
                        interactive=True,
                        placeholder="Onnx model output path.",
                    )
                with gr.Row():
                    infoOnnx = gr.Label(label="info")
                with gr.Row():
                    butOnnx = gr.Button(i18n("导出Onnx模型"), variant="primary")
                butOnnx.click(export_onnx, [ckpt_dir, onnx_dir], infoOnnx)

            with gr.TabItem(tab_faq):
                try:
                    with open(faq_file, "r", encoding="utf8") as f:
                        info = f.read()
                    gr.Markdown(value=info)
                except:
                    gr.Markdown(traceback.format_exc())

        # region Mangio Preset Handler Region
        def save_preset(
            preset_name,
            sid0,
            vc_transform,
            input_audio0,
            input_audio1,
            f0method,
            crepe_hop_length,
            filter_radius,
            file_index1,
            file_index2,
            index_rate,
            resample_sr,
            rms_mix_rate,
            protect,
            f0_file,
        ):
            data = None
            with open("../inference-presets.json", "r") as file:
                data = json.load(file)
            preset_json = {
                "name": preset_name,
                "model": sid0,
                "transpose": vc_transform,
                "audio_file": input_audio0,
                "auto_audio_file": input_audio1,
                "f0_method": f0method,
                "crepe_hop_length": crepe_hop_length,
                "median_filtering": filter_radius,
                "feature_path": file_index1,
                "auto_feature_path": file_index2,
                "search_feature_ratio": index_rate,
                "resample": resample_sr,
                "volume_envelope": rms_mix_rate,
                "protect_voiceless": protect,
                "f0_file_path": f0_file,
            }
            data["presets"].append(preset_json)
            with open("../inference-presets.json", "w") as file:
                json.dump(data, file)
                file.flush()
            print("Saved Preset %s into inference-presets.json!" % preset_name)

        def on_preset_changed(preset_name):
            print("Changed Preset to %s!" % preset_name)
            data = None
            with open("../inference-presets.json", "r") as file:
                data = json.load(file)

            print("Searching for " + preset_name)
            returning_preset = None
            for preset in data["presets"]:
                if preset["name"] == preset_name:
                    print("Found a preset")
                    returning_preset = preset
            # return all new input values
            return (
                # returning_preset['model'],
                # returning_preset['transpose'],
                # returning_preset['audio_file'],
                # returning_preset['f0_method'],
                # returning_preset['crepe_hop_length'],
                # returning_preset['median_filtering'],
                # returning_preset['feature_path'],
                # returning_preset['auto_feature_path'],
                # returning_preset['search_feature_ratio'],
                # returning_preset['resample'],
                # returning_preset['volume_envelope'],
                # returning_preset['protect_voiceless'],
                # returning_preset['f0_file_path']
            )

        # Preset State Changes

        # This click calls save_preset that saves the preset into inference-presets.json with the preset name
        # mangio_preset_save_btn.click(
        #     fn=save_preset,
        #     inputs=[
        #         mangio_preset_name_save,
        #         sid0,
        #         vc_transform0,
        #         input_audio0,
        #         f0method0,
        #         crepe_hop_length,
        #         filter_radius0,
        #         file_index1,
        #         file_index2,
        #         index_rate1,
        #         resample_sr0,
        #         rms_mix_rate0,
        #         protect0,
        #         f0_file
        #     ],
        #     outputs=[]
        # )

        # mangio_preset.change(
        #     on_preset_changed,
        #     inputs=[
        #         # Pass inputs here
        #         mangio_preset
        #     ],
        #     outputs=[
        #         # Pass Outputs here. These refer to the gradio elements that we want to directly change
        #         # sid0,
        #         # vc_transform0,
        #         # input_audio0,
        #         # f0method0,
        #         # crepe_hop_length,
        #         # filter_radius0,
        #         # file_index1,
        #         # file_index2,
        #         # index_rate1,
        #         # resample_sr0,
        #         # rms_mix_rate0,
        #         # protect0,
        #         # f0_file
        #     ]
        # )
        # endregion

        # with gr.TabItem(i18n("招募音高曲线前端编辑器")):
        #     gr.Markdown(value=i18n("加开发群联系我xxxxx"))
        # with gr.TabItem(i18n("点击查看交流、问题反馈群号")):
        #     gr.Markdown(value=i18n("xxxxx"))

        if (
            config.iscolab or config.paperspace
        ):  # Share gradio link for colab and paperspace (FORK FEATURE)
            app.queue(concurrency_count=511, max_size=1022).launch(share=True)
        else:
            app.queue(concurrency_count=511, max_size=1022).launch(
                server_name="0.0.0.0",
                inbrowser=not config.noautoopen,
                server_port=config.listen_port,
                quiet=False,
            )


# endregion

if __name__ == "__main__":
    GradioSetup()
