#[cfg(feature = "cuda")]
use candle_core::{Device, Tensor, Shape};
#[cfg(feature = "cuda")]
use vllm_cuda::PagedAttentionKernels;

#[test]
#[cfg(feature = "cuda")]
fn test_paged_attention_parity() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new_cuda(0)?;
    let candle_dev = match &device {
        Device::Cuda(dev) => dev,
        _ => panic!("Expected CUDA device"),
    };

    let kernels = PagedAttentionKernels::new(candle_dev, 0)?;

    let batch_size = 2;
    let num_heads = 4;
    let num_kv_heads = 4;
    let head_dim = 64;
    let block_size = 16;
    let max_blocks_per_seq = 64;
    let context_len = 512;
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Create inputs
    let q = Tensor::randn(0f32, 1f32, (batch_size, num_heads, head_dim), &device)?;
    let k_cache = Tensor::randn(0f32, 1f32, (max_blocks_per_seq * batch_size, num_kv_heads, block_size, head_dim), &device)?;
    let v_cache = Tensor::randn(0f32, 1f32, (max_blocks_per_seq * batch_size, num_kv_heads, block_size, head_dim), &device)?;
    
    // Block table: just sequential blocks for simplicity
    let mut bt_data = Vec::new();
    for i in 0..batch_size {
        for j in 0..max_blocks_per_seq {
            bt_data.push(i as i32 * max_blocks_per_seq as i32 + j as i32);
        }
    }
    let block_table = Tensor::from_vec(bt_data, (batch_size, max_blocks_per_seq), &device)?;
    
    let context_lens = Tensor::from_vec(vec![context_len as i32; batch_size], (batch_size,), &device)?;

    // Dummy rotation metadata for positional encoding
    let rotation_metadata = Tensor::zeros((batch_size, context_len), candle_core::DType::F32, &device)?;
    let (rm_storage, _) = rotation_metadata.storage_and_layout();


    let (q_storage, _) = q.storage_and_layout();
    let (k_storage, _) = k_cache.storage_and_layout();
    let (v_storage, _) = v_cache.storage_and_layout();
    let (bt_storage, _) = block_table.storage_and_layout();
    let (cl_storage, _) = context_lens.storage_and_layout();

    let q_cuda = match &*q_storage {
        candle_core::Storage::Cuda(c) => c,
        _ => panic!("q is not on CUDA"),
    };
    let k_cuda = match &*k_storage {
        candle_core::Storage::Cuda(c) => c,
        _ => panic!("k is not on CUDA"),
    };
    let v_cuda = match &*v_storage {
        candle_core::Storage::Cuda(c) => c,
        _ => panic!("v is not on CUDA"),
    };
    let bt_cuda = match &*bt_storage {
        candle_core::Storage::Cuda(c) => c,
        _ => panic!("bt is not on CUDA"),
    };
    let cl_cuda = match &*cl_storage {
        candle_core::Storage::Cuda(c) => c,
        _ => panic!("cl is not on CUDA"),
    };
    let rm_cuda = match &*rm_storage {
        candle_core::Storage::Cuda(c) => c,
        _ => panic!("rm is not on CUDA"),
    };

    let mut out_v1 = candle_dev.alloc_zeros::<f32>(batch_size * num_heads * head_dim)?;
    let mut out_v1_plus = candle_dev.alloc_zeros::<f32>(batch_size * num_heads * head_dim)?;
    let mut out_v2 = candle_dev.alloc_zeros::<f32>(batch_size * num_heads * head_dim)?;

    unsafe {
        // V1 Baseline
        kernels.launch_v1(
            q_cuda.as_cuda_slice::<f32>()?,
            k_cuda.as_cuda_slice::<f32>()?,
            v_cuda.as_cuda_slice::<f32>()?,
            bt_cuda.as_cuda_slice::<i32>()?,
            cl_cuda.as_cuda_slice::<i32>()?,
            &mut out_v1,
            scale,
            num_heads as i32,
            num_kv_heads as i32,
            head_dim as i32,
            block_size as i32,
            max_blocks_per_seq as i32,
            batch_size as i32,
        ).map_err(|e| format!("V1 launch failed: {:?}", e))?;

        // V1+ Optimized
        kernels.launch_v1_plus(
            q_cuda.as_cuda_slice::<f32>()?,
            k_cuda.as_cuda_slice::<f32>()?,
            v_cuda.as_cuda_slice::<f32>()?,
            bt_cuda.as_cuda_slice::<i32>()?,
            cl_cuda.as_cuda_slice::<i32>()?,
            &mut out_v1_plus,
            rm_cuda.as_cuda_slice::<f32>()?,
            scale,
            num_heads as i32,
            num_kv_heads as i32,
            head_dim as i32,
            block_size as i32,
            max_blocks_per_seq as i32,
            batch_size as i32,
        ).map_err(|e| format!("V1+ launch failed: {:?}", e))?;

        // V2 Partitioned/Reduced
        let num_partitions = (context_len as i32 + 255) / 256;
        let num_parts = num_partitions as usize;
        let tmp_out_size = batch_size * num_heads * num_parts * head_dim;
        let exp_sums_size = batch_size * num_heads * num_parts;
        let max_logits_size = batch_size * num_heads * num_parts;

        let mut tmp_out = candle_dev.alloc_zeros::<f32>(tmp_out_size)?;
        let mut exp_sums = candle_dev.alloc_zeros::<f32>(exp_sums_size)?;
        let mut max_logits = candle_dev.alloc_zeros::<f32>(max_logits_size)?;

        kernels.launch_v2_partition(
            q_cuda.as_cuda_slice::<f32>()?,
            k_cuda.as_cuda_slice::<f32>()?,
.as_cuda_slice::<f32>()?,
            bt_cuda.as_cuda_slice::<i32>()?,
            cl_cuda.as_cuda_slice::<i32>()?,
            &mut tmp_out,
            &mut exp_sums,
            &mut max_logits,
            rm_cuda.as_cuda_slice::<f32>()?,
            scale,
            num_heads as i32,
            num_kv_heads as i32,
            head_dim as i32,
            block_size as i32,
            max_blocks_per_seq as i32,
            num_partitions,
            batch_size as i32,
        ).map_err(|e| format!("V2 partition launch failed: {:?}", e))?;

        kernels.launch_v2_reduce(
            &mut out_v2,
            &exp_sums,
            &max_logits,
            &tmp_out,
            cl_cuda.as_cuda_slice::<i32>()?,
            rm_cuda.as_cuda_slice::<f32>()?,
            num_heads as i32,
            head_dim as i32,
            num_partitions,
            batch_size as i32,
        ).map_err(|e| format!("V2 reduce launch failed: {:?}", e))?;
    }

    // Download results
    let out_v1_tensor = Tensor::from_storage(
        candle_core::Storage::Cuda(candle_core::CudaStorage::wrap_cuda_slice(out_v1, candle_dev.clone())),
        Shape::from((batch_size, num_heads, head_dim)),
        candle_core::op::BackpropOp::none(),
        false,
    );
    let out_v1_plus_tensor = Tensor::from_storage(
        candle_core::Storage::Cuda(candle_core::CudaStorage::wrap_cuda_slice(out_v1_plus, candle_dev.clone())),
        Shape::from((batch_size, num_heads, head_dim)),
        candle_core::op::BackpropOp::none(),
        false,
    );
    let out_v2_tensor = Tensor::from_storage(
        candle_core::Storage::Cuda(candle_core::CudaStorage::wrap_cuda_slice(out_v2, candle_dev.clone())),
        Shape::from((batch_size, num_heads, head_dim)),
        candle_core::op::BackpropOp::none(),
        false,
    );

    let host_v1 = out_v1_tensor.flatten_all()?.to_vec1::<f32>()?;
    let host_v1_plus = out_v1_plus_tensor.flatten_all()?.to_vec1::<f32>()?;
    let host_v2 = out_v2_tensor.flatten_all()?.to_vec1::<f32>()?;

    // Compare
    let epsilon = 1e-4; // Slightly more relaxed for float accumulations
    for i in 0..host_v1.len() {
        let v1 = host_v1[i];
        let v1p = host_v1_plus[i];
        let v2 = host_v2[i];
        
        if (v1 - v1p).abs() > epsilon {
            panic!("V1 vs V1+ mismatch at index {}: {} vs {} (diff: {})", i, v1, v1p, (v1 - v1p).abs());
        }
        if (v1 - v2).abs() > epsilon {
            panic!("V1 vs V2 mismatch at index {}: {} vs {} (diff: {})", i, v1, v2, (v1 - v2).abs());
        }
    }

    println!("Success! All kernels produced identical results.");
    Ok(())
}
