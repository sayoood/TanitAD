# Top-20 kernels (self device time per step, RTX 4060, bs 1 fp32)

## Operative rollout fwd+bwd, K=15 (`components_fp32_bs1_trace.profiled_op_rollout`)

| # | kernel | calls/step | ms/step | avg us | % of kernel time |
|---|---|---|---|---|---|
| 1 | `ampere_sgemm_128x64_tn` | 391 | 190.7 | 487.8 | 23.5 |
| 2 | `ampere_sgemm_64x64_nt` | 285 | 145.7 | 511.2 | 17.9 |
| 3 | `_Z41fmha_cutlassB_f32_aligned_64x64_k128_sm80N22PyTorchMemEffAttention23AttentionBackwardKernelIN7cutlass4arch4Sm80EfLb1ELb0ELb0ELi64ELi64ELi128ELb0EE6ParamsE` | 90 | 134.0 | 1489.2 | 16.5 |
| 4 | `_ZN7cutlass7Kernel2I43cutlass_80_simt_sgemm_256x128_8x4_nn_align1EEvNT_6ParamsE` | 180 | 83.2 | 462.3 | 10.2 |
| 5 | `ampere_sgemm_128x64_nn` | 210 | 82.4 | 392.4 | 10.1 |
| 6 | `_ZN2at6native29vectorized_elementwise_kernelILi4ENS0_15CUDAFunctor_addIfEESt5arrayIPcLy3EEEEviT0_T1_` | 1776 | 56.1 | 31.6 | 6.9 |
| 7 | `_Z40fmha_cutlassF_f32_aligned_64x128_rf_sm80N22PyTorchMemEffAttention15AttentionKernelIfN7cutlass4arch4Sm80ELb1ELi64ELi128ELi128ELb1ELb1EE6ParamsE` | 90 | 49.7 | 552.7 | 6.1 |
| 8 | `ampere_sgemm_128x128_nt` | 105 | 16.8 | 159.7 | 2.1 |
| 9 | `_ZN2at6native29vectorized_elementwise_kernelILi4EZZZNS0_26GeluBackwardCUDAKernelImplERNS_18TensorIteratorBaseENS0_8GeluTypeEENKUlvE0_clEvENKUlvE0_clEvEUlffE_St5arrayIPcLy3EEEEviT0_T1_` | 105 | 12.3 | 117.2 | 1.5 |
| 10 | `_ZN2at6native53_GLOBAL__N__87a90936_20_layer_norm_kernel_cu_3ff0b71f39layer_norm_grad_input_kernel_vectorizedIffLb0EEEvPKT_S5_PKT0_S8_S5_PS3_i` | 195 | 6.9 | 35.2 | 0.8 |
| 11 | `_ZN2at6native13reduce_kernelILi128ELi4ENS0_8ReduceOpIfNS0_14func_wrapper_tIfZNS0_11sum_functorIfffEclERNS_14TensorIteratorEEUlffE_EEjfLi4ELi4EEEEEvT1_` | 451 | 5.8 | 12.9 | 0.7 |
| 12 | `_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_` | 270 | 4.3 | 15.8 | 0.5 |
| 13 | `_ZN2at6native29vectorized_elementwise_kernelILi4ENS0_11FillFunctorIfEESt5arrayIPcLy1EEEEviT0_T1_` | 272 | 3.7 | 13.4 | 0.4 |
| 14 | `Memcpy DtoD (Device -> Device)` | 270 | 3.1 | 11.5 | 0.4 |
| 15 | `_ZN2at6native29vectorized_elementwise_kernelILi4EZZZNS0_18GeluCUDAKernelImplERNS_18TensorIteratorBaseENS0_8GeluTypeEENKUlvE0_clEvENKUlvE0_clEvEUlfE_St5arrayIPcLy2EEEEviT0_T1_` | 105 | 2.8 | 26.9 | 0.3 |
| 16 | `ampere_sgemm_64x64_nn` | 1 | 2.3 | 2308.9 | 0.3 |
| 17 | `_ZN2at6native53_GLOBAL__N__87a90936_20_layer_norm_kernel_cu_3ff0b71f28vectorized_layer_norm_kernelIffLb0EEEviT0_PKT_S6_S6_PS3_S7_PS4_` | 195 | 2.2 | 11.2 | 0.3 |
| 18 | `_ZN7cutlass7Kernel2I43cutlass_80_simt_sgemm_128x128_8x4_nt_align1EEvNT_6ParamsE` | 1 | 2.0 | 1984.8 | 0.2 |
| 19 | `_ZN2at6native53_GLOBAL__N__87a90936_20_layer_norm_kernel_cu_3ff0b71f35GammaBetaBackwardCUDAKernelTemplateIffLj32ELj32ELj256ELb0ELb0ELb0EEEvxxPKT_S5_PKT0_S8_PS3_S9_` | 195 | 1.8 | 9.2 | 0.2 |
| 20 | `_ZN2at6native29vectorized_elementwise_kernelILi4ENS0_13BinaryFunctorIfffNS0_15binary_internal10MulFunctorIfEEEESt5arrayIPcLy3EEEEviT0_T1_` | 90 | 1.6 | 17.4 | 0.2 |

## Whole forward, no grad (`fwdprof_fp32_bs1_trace`)

| # | kernel | calls/step | ms/step | avg us | % of kernel time |
|---|---|---|---|---|---|
| 1 | `ampere_sgemm_128x64_tn` | 898 | 384.1 | 427.7 | 63.2 |
| 2 | `_Z40fmha_cutlassF_f32_aligned_64x128_rf_sm80N22PyTorchMemEffAttention15AttentionKernelIfN7cutlass4arch4Sm80ELb1ELi64ELi128ELi128ELb1ELb1EE6ParamsE` | 251 | 93.2 | 371.5 | 15.3 |
| 3 | `_ZN7cutlass7Kernel2I49cutlass_80_tensorop_d884gemm_32x64_16x4_nt_align1EEvNT_6ParamsE` | 1 | 42.1 | 42120.1 | 6.9 |
| 4 | `_Z10sytrd4_gpuI12sytrd_paramsIdLi32ELi8ELi1024ELi32ELi16ELi1ELi2EEEviPNT_9data_typeEiiPNS2_8pod_typeES4_iiiiS4_PjS4_S4_iii` | 16 | 13.6 | 849.3 | 2.2 |
| 5 | `ampere_sgemm_128x32_tn` | 258 | 9.8 | 37.8 | 1.6 |
| 6 | `_ZN2at6native29vectorized_elementwise_kernelILi4ENS0_15CUDAFunctor_addIfEESt5arrayIPcLy3EEEEviT0_T1_` | 571 | 7.9 | 13.8 | 1.3 |
| 7 | `_ZN2at6native29vectorized_elementwise_kernelILi4EZZZNS0_18GeluCUDAKernelImplERNS_18TensorIteratorBaseENS0_8GeluTypeEENKUlvE0_clEvENKUlvE0_clEvEUlfE_St5arrayIPcLy2EEEEviT0_T1_` | 282 | 6.5 | 23.0 | 1.1 |
| 8 | `_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_` | 285 | 6.4 | 22.5 | 1.1 |
| 9 | `_ZN2at6native53_GLOBAL__N__87a90936_20_layer_norm_kernel_cu_3ff0b71f28vectorized_layer_norm_kernelIffLb0EEEviT0_PKT_S6_S6_PS3_S7_PS4_` | 534 | 5.8 | 10.8 | 0.9 |
| 10 | `_Z10sytrd4_gpuI12sytrd_paramsIdLi32ELi8ELi512ELi32ELi16ELi1ELi2EEEviPNT_9data_typeEiiPNS2_8pod_typeES4_iiiiS4_PjS4_S4_iii` | 12 | 5.5 | 455.9 | 0.9 |
| 11 | `ampere_dgemm_64x64_lower_nt` | 56 | 5.4 | 97.1 | 0.9 |
| 12 | `_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastINS0_15CUDAFunctor_addIfEEEEvRNS_18TensorIteratorBaseERKT_EUliE_EEviT1_` | 30 | 5.3 | 177.6 | 0.9 |
| 13 | `_Z10sytrd4_ctaI12sytrd_paramsIdLi16ELi32ELi128ELi32ELi0ELi1ELi2EELi1EEviPNT_9data_typeEiiPNS2_8pod_typeEiS6_iS4_ii` | 1 | 2.6 | 2630.5 | 0.4 |
| 14 | `_Z9laed4_parI13stedc_params_I7double2dLi8ELi512ELi128EEEviiiPiPNT_9data_typeES6_iS6_S6_S6_iS3_i` | 2 | 2.6 | 1290.6 | 0.4 |
| 15 | `_ZN2at6native51_GLOBAL__N__67b3c33b_18_DepthwiseConv2d_cu_2ee6150b39conv_depthwise2d_forward_kernel_genericIfiEEvN5torch10headeronly6detail27GenericPackedTensorAccessorINS5_14TensorAccessorIN3c108ArrayRefIxEEKT_Ly3ENS4_16DefaultPtrTraitsEiEENS_6detail16IndexBoundsCheckILy4EiEESC_Ly4ESD_iEENS6_INS7_ISA_SB_Ly3ESD_iEESH_SB_Ly4ESD_iEESI_NS6_INS7_ISA_SC_Ly0ESD_iEENSG_ILy1EiEESC_Ly1ESD_iEEbT0_iiiiiiiiiiiiii` | 3 | 1.7 | 573.2 | 0.3 |
| 16 | `_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastINS0_13BinaryFunctorIfffNS0_15binary_internal10DivFunctorIfEEEEEEvRNS_18TensorIteratorBaseERKT_EUliE_EEviT1_` | 4 | 1.3 | 327.8 | 0.2 |
| 17 | `Memset (Device)` | 1193 | 1.0 | 0.8 | 0.2 |
| 18 | `_Z9laed4_parI13stedc_params_I7double2dLi8ELi256ELi128EEEviiiPiPNT_9data_typeES6_iS6_S6_S6_iS3_i` | 1 | 0.9 | 931.1 | 0.2 |
| 19 | `_ZN2at6native29vectorized_elementwise_kernelILi4EZZZNS0_15mse_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE_clEvENKUlvE0_clEvEUlffE_St5arrayIPcLy3EEEEviT0_T1_` | 2 | 0.9 | 462.4 | 0.2 |
| 20 | `_Z9laed4_parI13stedc_params_I7double2dLi8ELi128ELi128EEEviiiPiPNT_9data_typeES6_iS6_S6_S6_iS3_i` | 1 | 0.8 | 831.7 | 0.1 |
