# raw/sources.md — every code file and paper read for E-EGO-LIT-1 (2026-09-05)

All repository files were fetched from `raw.githubusercontent.com/<repo>/<full sha>/<path>` and hashed locally; the local pull lives at `C:/Users/Admin/ego_lit/code/<owner>_<repo>@<sha7>/` (dev box only — third-party code is NOT copied into this repo; cite `repo@sha:path:line`).

## Repositories (default branch HEAD at fetch time, 2026-09-05)

| repo | branch | commit | committed | subject |
|---|---|---|---|---|
| `hustvl/DiffusionDrive` | `main` | `9b52ed0ec06b073d82d6f392ab084c7b301c8681` | 2025-12-08 | Update README.md |
| `autonomousvision/carla_garage` | `leaderboard_2` | `72f39a63423a5edef6904b1487e0360a64bcf445` | 2026-06-22 | Merge pull request #117 from lloitesa013/fix/numpy-1.24-compat-bench2d |
| `jchengai/planTF` | `main` | `f54060850649710af8999c12d50598c8aeaa69aa` | 2024-07-11 | fix typos and correctly print simulation results |
| `jchengai/pluto` | `main` | `b9964b649c660f1f4a971d614c66f5992e24c18a` | 2024-07-15 | add training code |
| `NVlabs/Hydra-MDP` | `main` | `16f195c20f336c4590474dd2f0c711375b4d7a6a` | 2024-08-30 | Update README.md |
| `OpenDriveLab/UniAD` | `v2.0` | `609ee083ea51c3521c323f1279dfc4cee0e60467` | 2025-10-29 | Update README for UniAD 2.0 Release |
| `hustvl/VAD` | `main` | `1688c4b1c3a9e2e7873ca9700ff8058170c0e3c8` | 2026-01-31 | Update README.md |
| `wzzheng/GenAD` | `main` | `b16667f4a51612b577360c5df4fcdebc3e6f0d55` | 2025-05-07 | Delete projects/.DS_Store |
| `liyingyanUCAS/WoTE` | `main` | `298957c128a91d41a1c6075bd0bb6e7e845e093f` | 2025-06-29 | Update README.md |
| `Kguo-cs/iPad` | `main` | `78f990a20cf07d4eebe301399c1161ce7fdb4b59` | 2026-06-11 | Update download command for test dataset |
| `NVlabs/GTRS` | `master` | `92a740def80610e4096962d25cc837b21e72ef78` | 2026-02-09 | Fix bug in gather_traj.py |
| `William-Yao-2000/DriveSuprim` | `main` | `80fe792d7654a596d92e20d030d1650f6f605c02` | 2026-02-03 | fix bug |

Also read locally: `autonomousvision/navsim` devkit @ `0a380a9063d7162ec93d0f51e9990ebac585f720` (2025-10-27; clone at `C:/Users/Admin/navsim/devkit`) — `navsim/agents/transfuser/transfuser_features.py`, `transfuser_model.py`, `navsim/agents/ego_status_mlp_agent.py`.

## Files (sha256 of the fetched bytes)

| repo@sha7 | path | bytes | sha256 |
|---|---|---|---|
| `Kguo-cs/iPad@78f990a` | `README.md` | 3334 | `5a5002b8cb356b1f7f4f5218574340f25c77bc75c5b8436fda04f3c9e3f512ee` |
| `Kguo-cs/iPad@78f990a` | `navsim/agents/pad/b2d_config.py` | 1329 | `31f2d25fbfabab724d4b8f047fcc659b302cb972f3c439eee038ef7b4620d7b0` |
| `Kguo-cs/iPad@78f990a` | `navsim/agents/pad/navsim_config.py` | 4384 | `3338fbd75c7c8a0c0976d705282d9b0daee14e9b1abcecbcca655c7ed476d07f` |
| `Kguo-cs/iPad@78f990a` | `navsim/agents/pad/pad_agent.py` | 18015 | `1753be8b2f4d6ed27cf534b41593a697cad62c583eca01639828810a3c0f465f` |
| `Kguo-cs/iPad@78f990a` | `navsim/agents/pad/pad_features.py` | 15050 | `fb17aa4225d4ebe3fedf0aaadbd73e13d0ead2609c5b7d1e1876651f0225dc5a` |
| `Kguo-cs/iPad@78f990a` | `navsim/agents/pad/pad_model.py` | 2663 | `9f9738ea3fe26a429eb2298461944412b6482d7406d8133f56acedd6d0575118` |
| `Kguo-cs/iPad@78f990a` | `navsim/agents/pad/score_module/scorer.py` | 2707 | `99d507dbb28d4aea3432ebb7a75590794dc23169605ff81cc4daa52f06c14049` |
| `Kguo-cs/iPad@78f990a` | `navsim/agents/pad/traj_refiner.py` | 1131 | `7d05e7e3ef480256ffe88f1931be2255ff7cd1d1de682d452bdc8db927da7561` |
| `NVlabs/GTRS@92a740d` | `README.md` | 3427 | `fa54f261badf8da6c683c49c467beee7c7f8e1cd35bcea57f352fc3ea261f008` |
| `NVlabs/GTRS@92a740d` | `navsim/agents/dp/dp_agent.py` | 6072 | `cf090412897e9adf86097a8544bce1d65b776416955f0d0d2d5644b0b8861f23` |
| `NVlabs/GTRS@92a740d` | `navsim/agents/dp/dp_config.py` | 6082 | `70c81fffd23446d1db3ec70ae6fc4cc44e8d51f35050a778c0c467b0b3e836a2` |
| `NVlabs/GTRS@92a740d` | `navsim/agents/dp/dp_model.py` | 15326 | `952fa38031919fe42d1d4fcbe4b42cb5c44c160afdf0f9e75d50efce9440d66d` |
| `NVlabs/GTRS@92a740d` | `navsim/agents/gtrs_dense/gtrs_agent.py` | 12257 | `f137b1d4544b5abcb972255014f59cb52ea1af4c711e796df06cd263d736430c` |
| `NVlabs/GTRS@92a740d` | `navsim/agents/gtrs_dense/hydra_backbone.py` | 2021 | `7727be7d1b6986ba96097d488fb34a6f4522ef85fe4a378436d08c98fb00eca6` |
| `NVlabs/GTRS@92a740d` | `navsim/agents/gtrs_dense/hydra_backbone_bev.py` | 4926 | `e5ee5891df0d7e262c8eb5db2db4cafbbb635298dcde73db013d15f8a0a050b6` |
| `NVlabs/GTRS@92a740d` | `navsim/agents/gtrs_dense/hydra_config.py` | 5989 | `a851aeb46dd3d6a100dab39d2aa2b3ade0834fead8a75d4c1b5304b74925f896` |
| `NVlabs/GTRS@92a740d` | `navsim/agents/gtrs_dense/hydra_features.py` | 20178 | `0d22691b058672f3e3f8ac0a5d49e2f89253bc0100258c98f43f295df2e0a41e` |
| `NVlabs/GTRS@92a740d` | `navsim/agents/gtrs_dense/hydra_model.py` | 14778 | `f8abb025a2a73d1ff480bc63a45b053b03c03023d41d4ecf052275e295e69af3` |
| `NVlabs/GTRS@92a740d` | `navsim/agents/tools/gen_vocab_score.py` | 6242 | `f673eefe6cabf2c24620756d021ffc3ef6ba8c65c6d2e5dd438d1ef48693901f` |
| `OpenDriveLab/UniAD@609ee08` | `projects/configs/stage2_e2e/base_e2e.py` | 24161 | `2d45be9b330f8e82456203e2d6374053bd7b80171c5316412dfa88b32e187b8c` |
| `OpenDriveLab/UniAD@609ee08` | `projects/mmdet3d_plugin/uniad/dense_heads/planning_head.py` | 11392 | `1c851bbb30a89a472632e1f6428910fbc6d24aa324854c790e6dd155c01cd16b` |
| `OpenDriveLab/UniAD@609ee08` | `projects/mmdet3d_plugin/uniad/detectors/uniad_e2e.py` | 18006 | `3040148bb5e03a88e02ff862ce5373c1aa5c75c14b497d535296bf18f05d32aa` |
| `OpenDriveLab/UniAD@609ee08` | `projects/mmdet3d_plugin/uniad/modules/transformer.py` | 13246 | `38a447652d55eb873e1d869c99591aaec766f2692ad85357d9f2e3a8d8a3915b` |
| `William-Yao-2000/DriveSuprim@80fe792` | `README.md` | 3265 | `563b3738bdd3a5bb25089108bf55345996444ac9e2cbc10746db5a9d15976fb3` |
| `William-Yao-2000/DriveSuprim@80fe792` | `navsim/agents/drivesuprim/data/transforms.py` | 729 | `1856280d28949be45aa18d81e0cb008f0412afa866ef7d7770bc09cd446f7f0d` |
| `William-Yao-2000/DriveSuprim@80fe792` | `navsim/agents/drivesuprim/drivesuprim_agent.py` | 15154 | `6645e12dfbaf4e1cc384b2b47861f3dbe7d1676ed0013185a19631ff209e5593` |
| `William-Yao-2000/DriveSuprim@80fe792` | `navsim/agents/drivesuprim/drivesuprim_backbone_pe.py` | 5140 | `b623ab33dc65cac0f8b349a8d4ff1dc59594eb04b491b0db705d8c639e5ba865` |
| `William-Yao-2000/DriveSuprim@80fe792` | `navsim/agents/drivesuprim/drivesuprim_config.py` | 3039 | `0d2ef14ff33d8bfab3f99443e8b0a3e0768cba06c7a651303bf2ba0cae3840d8` |
| `William-Yao-2000/DriveSuprim@80fe792` | `navsim/agents/drivesuprim/drivesuprim_features.py` | 11628 | `82b9454985887a389f3d025d37214dad83f4160f637eca5028b867be60ed4086` |
| `William-Yao-2000/DriveSuprim@80fe792` | `navsim/agents/drivesuprim/drivesuprim_loss_fn.py` | 9056 | `15d489316eac0e5c461c937ab309c1ae890177df9e0cd0affcda4352a0170896` |
| `William-Yao-2000/DriveSuprim@80fe792` | `navsim/agents/drivesuprim/drivesuprim_model.py` | 17790 | `3a069b5befbc313b7881745641e0b1154d90c222462e1d6e8e971b810727ed4b` |
| `William-Yao-2000/DriveSuprim@80fe792` | `navsim/agents/drivesuprim/ssl_meta_arch.py` | 3283 | `a239f83e9496a32cc2e3dad7f1799c3fb42544a2b5f1ad111f982fb1ac28924d` |
| `William-Yao-2000/DriveSuprim@80fe792` | `navsim/agents/drivesuprim/utils/distributed.py` | 8576 | `b6aaa4151999c88a4ff031e786dc66bb2ecbc8b5421ffe72169c9c305c1414e5` |
| `autonomousvision/carla_garage@72f39a6` | `team_code/aim.py` | 2583 | `2845031526d7be859352cfaa2a842628183c9a53cf7e4e210a505811f7ceea29` |
| `autonomousvision/carla_garage@72f39a6` | `team_code/config.py` | 45039 | `46e0c91128a5e26184fea52a012681b084a7e35dc62638eab5709b9a48733a51` |
| `autonomousvision/carla_garage@72f39a6` | `team_code/data.py` | 57652 | `96a4c440b5d2613a9bdbb85299e09df94921b71251a225e7a6417df591f0872a` |
| `autonomousvision/carla_garage@72f39a6` | `team_code/model.py` | 46994 | `9cd5352354ff223d7858e4c39accc6b3d0b58e500a607631747b956dfd61db76` |
| `autonomousvision/carla_garage@72f39a6` | `team_code/sensor_agent.py` | 39616 | `14dfe25b7d0d7444f8ac48df30cfc96722ba52b12a09ff627786da3deafcfbbe` |
| `autonomousvision/carla_garage@72f39a6` | `team_code/transfuser.py` | 21121 | `f0da9b26095b228fe955076b34012472f73086204ed76b34c21e9b3cab5d67b8` |
| `hustvl/DiffusionDrive@9b52ed0` | `navsim/agents/diffusiondrive/modules/blocks.py` | 4257 | `de879c63cbfaf8fe96437cb84f4a9c28392625ec6706132032afceed4ffc2776` |
| `hustvl/DiffusionDrive@9b52ed0` | `navsim/agents/diffusiondrive/modules/conditional_unet1d.py` | 10056 | `0d6d589fb29d6e076b77bddbb3d7ab6b86c216c29e13391fb532d661a13f5cf7` |
| `hustvl/DiffusionDrive@9b52ed0` | `navsim/agents/diffusiondrive/modules/multimodal_loss.py` | 6757 | `699d732b1c8d858ed8b5cd72f31607ed433e031c12441c46baac697a463283fd` |
| `hustvl/DiffusionDrive@9b52ed0` | `navsim/agents/diffusiondrive/modules/scheduler.py` | 1747 | `5a602966f2f5c923c504ce1108129c94995d0aaa45204d4910254b630e042aaa` |
| `hustvl/DiffusionDrive@9b52ed0` | `navsim/agents/diffusiondrive/transfuser_agent.py` | 7661 | `263239dd8f4a647e2d2be0856c62d64ff6806e6923961f9aabb24510726cf6c6` |
| `hustvl/DiffusionDrive@9b52ed0` | `navsim/agents/diffusiondrive/transfuser_backbone.py` | 19896 | `53cb334652860634cf2cd9c4867b270465631f5f1e33b2e4c7bcadfc448471a3` |
| `hustvl/DiffusionDrive@9b52ed0` | `navsim/agents/diffusiondrive/transfuser_callback.py` | 12482 | `e56c1af7d9bafbf994602727719c4579462757005cfe500ce8482567cc06249b` |
| `hustvl/DiffusionDrive@9b52ed0` | `navsim/agents/diffusiondrive/transfuser_config.py` | 4520 | `9be65fcc34a7ea9d2d1662adf70e3aa6c4807faf049c168956f3e12adb25cefb` |
| `hustvl/DiffusionDrive@9b52ed0` | `navsim/agents/diffusiondrive/transfuser_features.py` | 15827 | `4b80250c498b53396c8515cc0f5927f8f71aed9b6825a750ff440431af2ff6c3` |
| `hustvl/DiffusionDrive@9b52ed0` | `navsim/agents/diffusiondrive/transfuser_loss.py` | 6486 | `5b70e51da6ac10d3b1f2c805db689b0ff8fe738d960a224ea578c19d529d7790` |
| `hustvl/DiffusionDrive@9b52ed0` | `navsim/agents/diffusiondrive/transfuser_model_v2.py` | 22659 | `84a7f824fa51f172d6d2a940ad566a96eb253f26231065f5283bf6507701a697` |
| `hustvl/VAD@1688c4b` | `projects/configs/VAD/VAD_base_e2e.py` | 17089 | `7b4460a6c65985b16d8c58cb18d253efd25b65c043a0369ef978c91460c514a5` |
| `hustvl/VAD@1688c4b` | `projects/mmdet3d_plugin/VAD/VAD.py` | 27020 | `e50cb55c6894ea386168e5e91c0810dc427414d71f769d25182af8410d6b892e` |
| `hustvl/VAD@1688c4b` | `projects/mmdet3d_plugin/VAD/VAD_head.py` | 91044 | `4e3c828125aeee521fe40756a422b5fb13410381c2dfd998ab26048f75b7f22f` |
| `hustvl/VAD@1688c4b` | `projects/mmdet3d_plugin/VAD/modules/transformer.py` | 11984 | `971f7096db501866250809995ae8775996c6baf7b2201876c719118747687a4a` |
| `hustvl/VAD@1688c4b` | `projects/mmdet3d_plugin/datasets/nuscenes_vad_dataset.py` | 85074 | `303745ced08745eb4f4136bf76a9c23ee4757677ad866151d1b8d4d1a8493ed5` |
| `hustvl/VAD@1688c4b` | `tools/data_converter/vad_nuscenes_converter.py` | 42317 | `258c7a63fde1dba8877910a488037a423c9c6cdba8f99e2ee02cc012a32c1460` |
| `jchengai/planTF@f540608` | `config/custom_trainer/planTF.yaml` | 63 | `0b17ada3e368db25ae121ba77b373c1c5133a88440fca586257c194c6c79040f` |
| `jchengai/planTF@f540608` | `config/data_augmentation/state_perturbation.yaml` | 372 | `87a0af823e2a4188ecf8b5c5f568810d0a44a74ff40a3c5040e2d97eec9bc966` |
| `jchengai/planTF@f540608` | `config/default_nuboard.yaml` | 859 | `cc50da1d7a537547ff36d841897e8a3c0136491a29b4a320846615fd3eed4617` |
| `jchengai/planTF@f540608` | `config/default_simulation.yaml` | 3738 | `02db262e89c7299de04a17f0ac603c21adcdd6d402e89fcda7542dce6db2bc6f` |
| `jchengai/planTF@f540608` | `config/default_training.yaml` | 2155 | `d4a782e78a431ea578147e20dfa412a7c3ced76f2d4566ed955756b157563c54` |
| `jchengai/planTF@f540608` | `config/lightning/custom_lightning.yaml` | 2341 | `f5c59707aeaa5a2da8ee30bcf49444b6a0263582f2fcbc240637280ecec43ae2` |
| `jchengai/planTF@f540608` | `config/model/planTF.yaml` | 509 | `c3ad1d7af3270ca786cb33ee6958e76cf0ec805a8648a2fe6ed949b2da16bedd` |
| `jchengai/planTF@f540608` | `config/planner/planTF.yaml` | 750 | `e7efe789943b6f3634175494420176daa5a1e96b5b6ad461a21433f241c09946` |
| `jchengai/planTF@f540608` | `config/scenario_filter/mini.yaml` | 1494 | `72c7e26a0deeb574f2301ac248a21d4df770bedbd49db41e2ea7b69443d89480` |
| `jchengai/planTF@f540608` | `config/scenario_filter/single_right_turn.yaml` | 1358 | `0993ecc66e918c9a4e41206ec24e51171c2abc7ff7fa704697df33d1454793e7` |
| `jchengai/planTF@f540608` | `config/scenario_filter/test14-hard.yaml` | 6900 | `4a8b8ba533f7c08855b1b06446c9ed94bf4f83826667a265d95b60496a1655c0` |
| `jchengai/planTF@f540608` | `config/scenario_filter/test14-random.yaml` | 1730 | `a0c48d6b66ee3460317c881411fd445b544306583a1a93ab855bcde0fdf77b9c` |
| `jchengai/planTF@f540608` | `config/scenario_filter/training_scenarios_1M.yaml` | 1498 | `48e65ed2b6401c598a6e4e80f0ece294c0c9ca80b3e419e696caa2c29479d528` |
| `jchengai/planTF@f540608` | `config/scenario_filter/val14.yaml` | 26912 | `18bc1fef68d42447664ce60282cc3467126beb3b7f1ed5c7830f4aa329af5818` |
| `jchengai/planTF@f540608` | `config/training/train_planTF.yaml` | 345 | `56f30ff27b0e498d2f2058aa9646d793a95537e4ef24c67422473e6ee2e87dbd` |
| `jchengai/planTF@f540608` | `src/data_augmentation/state_perturbation.py` | 5953 | `d248ef6f2d7f94e7cf94f0c31b5042bd814f330c2dab6c2555b1faea9dd8d6ea` |
| `jchengai/planTF@f540608` | `src/feature_builders/nuplan_feature_builder.py` | 18781 | `270d4d863aa5159e5fb9d6841e595aca64f997f8b48f9a4694a79f1bdbaad1a4` |
| `jchengai/planTF@f540608` | `src/features/nuplan_feature.py` | 4660 | `081883870dfc09ff05f86948572b040185ece4b43c2ee438abbe08a8e74dfac8` |
| `jchengai/planTF@f540608` | `src/models/planTF/layers/common_layers.py` | 1181 | `3a4132f0f544b9cab5a401b8fcf7b7acd74cb3306484258337d1c9838e41b879` |
| `jchengai/planTF@f540608` | `src/models/planTF/layers/embedding.py` | 8550 | `553e5ba48f4c5cf7a0c01197340bc84c3168dd3f886a2ee04d0abc6fa82608ef` |
| `jchengai/planTF@f540608` | `src/models/planTF/layers/transformer_encoder_layer.py` | 2332 | `57d8aeb9cc1c56db7ceeb0274ce382aca7504dd205a19cf55f42a67f074bfddb` |
| `jchengai/planTF@f540608` | `src/models/planTF/lightning_trainer.py` | 8692 | `73f9e8ed16556390ce2ec16e4011897e2a0191c948a10c487b99bb18e8abac7a` |
| `jchengai/planTF@f540608` | `src/models/planTF/modules/agent_encoder.py` | 4803 | `844c0896b34710341d0b2b0df7e3bcb9fdd37896d5a87b51fc06c17aa5e8c09e` |
| `jchengai/planTF@f540608` | `src/models/planTF/modules/map_encoder.py` | 2552 | `f286aa2ccd5345ccf12da082eeabdef05a5fb17da4dd986531fa7e91e77c3ab0` |
| `jchengai/planTF@f540608` | `src/models/planTF/modules/trajectory_decoder.py` | 1131 | `60f038ae47058842ddc9287f4093508e819602126f3120519a17007fcf982930` |
| `jchengai/planTF@f540608` | `src/models/planTF/planning_model.py` | 5184 | `066fb164adefa6e6aee9c4edadf28b60dfb4a74c6ba9b498fcfddd586344f23c` |
| `jchengai/pluto@b9964b6` | `config/custom_trainer/pluto_trainer.yaml` | 57 | `8bf1b306e9270b8127382cb93a35362a4c73229968a032a7541f738b4cd34be4` |
| `jchengai/pluto@b9964b6` | `config/data_augmentation/contrastive_scenario_generator.yaml` | 284 | `74c0c8739ca33dd7ea0a490a1a89e0d57bafd3c1689f30686fe20e3b841bd83b` |
| `jchengai/pluto@b9964b6` | `config/default_simulation.yaml` | 3738 | `02db262e89c7299de04a17f0ac603c21adcdd6d402e89fcda7542dce6db2bc6f` |
| `jchengai/pluto@b9964b6` | `config/default_training.yaml` | 2369 | `01c69b1342f192a7cf64ab57aacad2f07d489f2d51ac556ba79f979efd6890d8` |
| `jchengai/pluto@b9964b6` | `config/lightning/custom_lightning.yaml` | 2341 | `f5c59707aeaa5a2da8ee30bcf49444b6a0263582f2fcbc240637280ecec43ae2` |
| `jchengai/pluto@b9964b6` | `config/model/pluto_model.yaml` | 586 | `069a48aee4455b2ba0c4173f6ad391e978e511acb9d1d20bd08adf8122e28271` |
| `jchengai/pluto@b9964b6` | `config/planner/pluto_planner.yaml` | 994 | `4af2618b635eec4e9fbb59034395eba13401ef2d271078484cb431227c83f590` |
| `jchengai/pluto@b9964b6` | `config/scenario_filter/mini_demo_scenario.yaml` | 1486 | `1c55c29584252897efbca53a80974cdc1c896c00da8ed37ab86bb9b7803c433c` |
| `jchengai/pluto@b9964b6` | `config/scenario_filter/random14_benchmark.yaml` | 6360 | `84db9dc656e71f9f85cc5d81eafd0127bd7a2e9a3e4d4f8f641536e8f58d15c9` |
| `jchengai/pluto@b9964b6` | `config/scenario_filter/training_scenarios_1M.yaml` | 1498 | `6bce7d2e26604a81b52dad818ef3394b2383d15ae5d4216ad4e71cca3ccbf01f` |
| `jchengai/pluto@b9964b6` | `config/scenario_filter/training_scenarios_tiny.yaml` | 1492 | `eb3a3be701620c3335b2340c3b78e285e3cbfbfb48e21b24a39d0c3216d41f6a` |
| `jchengai/pluto@b9964b6` | `config/scenario_filter/val14_benchmark.yaml` | 22828 | `386432740d38a4b20a38fa761cfe9d7a32f24ade1f40824596e05f2beefe61c2` |
| `jchengai/pluto@b9964b6` | `config/scenario_filter/val_demo_scenario.yaml` | 1359 | `2d0fd54dd29874b819182f5fd583c21baa1ca2caa207a6844ecb82e912132ccc` |
| `jchengai/pluto@b9964b6` | `config/training/train_pluto.yaml` | 418 | `99102a2708b674576cb9d442cddbe05e1b0f8362e64bee644979f6d504ec64ed` |
| `jchengai/pluto@b9964b6` | `src/data_augmentation/contrastive_scenario_generator.py` | 12990 | `8dd2c6623414f4bc029f297e317d1f33f24f459fa765c3738c1526f1fdae1fce` |
| `jchengai/pluto@b9964b6` | `src/feature_builders/pluto_feature_builder.py` | 32682 | `fe2e988d809dc907ca2a47d3572959a08b4bc18d76bed1cc78dce77f10285063` |
| `jchengai/pluto@b9964b6` | `src/models/pluto/layers/embedding.py` | 8550 | `553e5ba48f4c5cf7a0c01197340bc84c3168dd3f886a2ee04d0abc6fa82608ef` |
| `jchengai/pluto@b9964b6` | `src/models/pluto/layers/fourier_embedding.py` | 2089 | `d03835d87ec40d9a075ab36f59e35dae0cd3e21a39aaec4d72d48caf3bb8194d` |
| `jchengai/pluto@b9964b6` | `src/models/pluto/modules/agent_encoder.py` | 4803 | `844c0896b34710341d0b2b0df7e3bcb9fdd37896d5a87b51fc06c17aa5e8c09e` |
| `jchengai/pluto@b9964b6` | `src/models/pluto/modules/agent_predictor.py` | 839 | `c00249a3c7dbe272264f4815a4b0a9e0ed417307906d5465f36a4d23922c1118` |
| `jchengai/pluto@b9964b6` | `src/models/pluto/modules/map_encoder.py` | 3505 | `855f6042543335b03d718881f16e7a8d42c4feecf44fac66c3201d60fe3b6ee5` |
| `jchengai/pluto@b9964b6` | `src/models/pluto/modules/planning_decoder.py` | 6129 | `0e7d65ba875ff075248f37ea8143540907df209d89b6566848303478ea721db1` |
| `jchengai/pluto@b9964b6` | `src/models/pluto/modules/static_objects_encoder.py` | 1046 | `8c06aee6769a2e4f2dee3fb39ab5185c2371ccbfad133bcc428ed833a297a9fa` |
| `jchengai/pluto@b9964b6` | `src/models/pluto/pluto_model.py` | 8560 | `d66b2efb35dfaa1f9453dd30ce8a84fc37fe8ee970fd8dfcffa65a7bd11112b8` |
| `jchengai/pluto@b9964b6` | `src/models/pluto/pluto_trainer.py` | 16888 | `1214b85636df230b61957ca5f5111c8897e00cfd6352e59e32b3b4a0f2a2505c` |
| `liyingyanUCAS/WoTE@298957c` | `README.md` | 3693 | `35bd054d55a514cb1b763947fa1ae07242491b90ee36a1ff93cd655655012bd2` |
| `liyingyanUCAS/WoTE@298957c` | `README_navsim.md` | 6543 | `28039487ec0439f7ede11d5ed2ac65c2d0f9b54157147fe5ae74be368235e615` |
| `liyingyanUCAS/WoTE@298957c` | `navsim/agents/WoTE/WoTE_agent.py` | 8328 | `d7dec084057780a1b265057841756bdf9cb4836c6a20a08776679f7a1b886537` |
| `liyingyanUCAS/WoTE@298957c` | `navsim/agents/WoTE/WoTE_features.py` | 4398 | `93a014d6e89b57251b24b1c6c215213165c769aceaf93737920a03761d8b26c8` |
| `liyingyanUCAS/WoTE@298957c` | `navsim/agents/WoTE/WoTE_loss.py` | 14708 | `2e3e13847a362a37282e5e6d6d45a3923eef33383efeee7f0a8db176248e9c6d` |
| `liyingyanUCAS/WoTE@298957c` | `navsim/agents/WoTE/WoTE_model.py` | 41430 | `206d0b548a5883ebf0ab3bc8191c92facc8a769da53def252e177bd03ae4e58d` |
| `liyingyanUCAS/WoTE@298957c` | `navsim/agents/WoTE/WoTE_targets.py` | 20441 | `5430bb1cbf0409533963c15595400ddbd000cc3001a1f3cb939379d8553f0e19` |
| `liyingyanUCAS/WoTE@298957c` | `navsim/agents/WoTE/__init__.py` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `liyingyanUCAS/WoTE@298957c` | `navsim/agents/WoTE/configs/default.py` | 4929 | `7c3673031822a11747528d76084861d96edc7d69e1222311853d6c8544cdae09` |
| `liyingyanUCAS/WoTE@298957c` | `navsim/agents/WoTE/modules/blocks.py` | 4257 | `de879c63cbfaf8fe96437cb84f4a9c28392625ec6706132032afceed4ffc2776` |
| `liyingyanUCAS/WoTE@298957c` | `navsim/agents/WoTE/modules/multimodal_loss.py` | 6558 | `f95c3e8a362d12f474371e73f866cc370e741e83a620b6c3376678b5b19efa76` |
| `wzzheng/GenAD@b16667f` | `projects/configs/GenAD/GenAD_config.py` | 17360 | `69db9878ba36f4088a788104250a62508f045dc97cca8d98297238922e2caed2` |
| `wzzheng/GenAD@b16667f` | `projects/mmdet3d_plugin/GenAD/GenAD.py` | 28439 | `74a807bc5363c9622cc1dedd90b499dd1259fef478048a0a3589aeb0a2ff9b2c` |
| `wzzheng/GenAD@b16667f` | `projects/mmdet3d_plugin/GenAD/GenAD_head.py` | 99777 | `b535a2943dda0a76811df2493e33dcaf7fd4ec79dd238bc6eddcd8174f442b67` |
| `wzzheng/GenAD@b16667f` | `projects/mmdet3d_plugin/GenAD/GenAD_transformer.py` | 19890 | `677ee492e48f7a5659842cb16c224ee6be55c4a924b2fe448e069698b8844509` |
| `wzzheng/GenAD@b16667f` | `projects/mmdet3d_plugin/GenAD/generator/state_prediction.py` | 1254 | `861d14ca3851eeda3019cbcb80730c4d0df651a53d2b4cc3836c16d58a593db0` |
| `wzzheng/GenAD@b16667f` | `projects/mmdet3d_plugin/GenAD/modules/transformer.py` | 11984 | `971f7096db501866250809995ae8775996c6baf7b2201876c719118747687a4a` |

## Papers (all banked in `TanitAD Research Lab/Library/`; library keys)

| key | paper |
|---|---|
| `2411.15139` | DiffusionDrive (CVPR 2025) — REF-C's decoder ancestor |
| `2205.15997` | TransFuser (PAMI 2022) |
| `2306.07957` | Hidden Biases of End-to-End Driving Models (TransFuser++, ICCV 2023) |
| `2406.15349` | NAVSIM (NeurIPS 2024 D&B) |
| `2309.10443` | PlanTF — Rethinking Imitation-based Planners (ICRA 2024) |
| `2404.14327` | PLUTO |
| `2408.03601` | DRAMA |
| `2406.06978` | Hydra-MDP |
| `2503.12820` | Hydra-MDP++ |
| `2212.10156` | UniAD (CVPR 2023) |
| `2303.12077` | VAD (ICCV 2023) |
| `2402.13243` | VADv2 |
| `2402.11502` | GenAD (ECCV 2024) |
| `2504.01941` | WoTE |
| `2505.15111` | iPad |
| `2506.06659` | DriveSuprim (AAAI 2026) |
| `2510.17191` | SimpleVSF |
| `2506.06664` | GTRS — Generalized Trajectory Scoring (banked by this WP) |
| `2405.19620` | SparseDrive — DiffusionDrive's nuScenes host |
| `2306.07962` | PDM — Parting with Misconceptions (CoRL 2023) |
| `2312.03031` | BEV-Planner — Is Ego Status All You Need? |
| `paradrive-cvpr2024` | PARA-Drive (local bank, CVF PDF) |
| `2206.08129` | TCP — REF-C's measurement-encoder ancestor |
| `2511.13079` | AdaptiveAD |
| `2207.09705` | Residual action prediction |

Text extraction for reading: `pdftotext -layout` on the banked PDFs, local copies at `C:/Users/Admin/ego_lit/papers/<key>.txt` (+ `<key>.hits.txt` keyword passages). Every number quoted in RESULT.md was read from the extracted text of the banked PDF or from the pinned source file named beside it.

## TanitAD measured inputs (ours)

- `C:/Users/Admin/refcv4b_egodrop/EGODROP_9500.json` (refcv4b step 9,500, ckpt md5 `ab4cd79e35b92bf70d6e2b80ba053e51`, 4,823 windows / 141 episodes, paired episode-cluster bootstrap n_boot 2000 seed 0) — being banked by the sibling WP `…/2026-09-05-refcv4b-egodrop-burden/`; `EGODROP_5000.json` (step 5,000) beside it.
- `stack/tanitad/refs/refc.py` md5 `547779023248eb6bb36be940fca73dae`, `refc_v3.py` `815fdd9aee19289fe04dd591aa85b28c` (off-Drive mirror `C:/Users/Admin/tanitad-wt`, md5-identical to the 2026-09-05 audit's worktree), `stack/scripts/refc_v3_train.py` (mirror).
