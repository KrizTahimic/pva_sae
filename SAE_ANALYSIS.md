Complete SAE Analysis Implementation Guide

This come from the code of the inspiration paper "Do I Know This Entity?" paper by Javier 2024

  Core Architecture Overview

  This implementation follows a three-stage pipeline:
  1. Activation Collection: Memory-mapped caching of transformer activations at specific token positions
  2. SAE Feature Analysis: Loading SAEs and computing feature activation differences between known/unknown entities
  3. Validation & Intervention: Statistical filtering and steering experiments

  ---
  1. ACTIVATION COLLECTION - THE FOUNDATION

  1.1 Hook-Based Activation Extraction System

  Core Hook Function (utils/activation_cache.py:34-39):
  def _get_activations_pre_hook(cache: Float[Tensor, "pos d_model"]):
      def hook_fn(module, input):
          nonlocal cache
          # Extract raw activations BEFORE they pass through the transformer block
          activation: Float[Tensor, "batch_size seq_len d_model"] = input[0].clone().to(cache)
          # Accumulate activations in pre-allocated memory
          cache[:, :] += activation[:, :].to(cache)
      return hook_fn

  WHY PRE-HOOKS: They capture activations before any layer processing (attention/MLP), giving clean residual stream representations that SAEs are
  trained on.

  1.2 Fixed-Length Activation Collection

  Complete Implementation (utils/activation_cache.py:42-80):
  @torch.no_grad()
  def _get_activations_fixed_seq_len(
      model, tokenizer, prompts: List[str], 
      block_modules: List[torch.nn.Module], 
      seq_len: int = 512, 
      layers: List[int] = None, 
      batch_size = 32, 
      save_device: Union[torch.device, str] = "cuda", 
      verbose = True
  ) -> Tuple[Float[Tensor, 'n seq_len'], Float[Tensor, 'n layer seq_len d_model']]:

      torch.cuda.empty_cache()

      if layers is None:
          layers = range(model.config.num_hidden_layers)

      n_layers = len(layers)
      d_model = model.config.hidden_size

      # PRE-ALLOCATE HIGH-PRECISION TENSORS
      # Shape: (num_prompts, num_layers, sequence_length, hidden_dim)
      activations = torch.zeros((len(prompts), n_layers, seq_len, d_model), device=save_device)
      all_input_ids = torch.zeros((len(prompts), seq_len), dtype=torch.long, device=save_device)

      # Initialize with pad tokens for proper masking
      all_input_ids.fill_(tokenizer.pad_token_id)

      # BATCH PROCESSING LOOP
      for i in tqdm(range(0, len(prompts), batch_size), disable=not verbose):
          # Tokenize current batch
          inputs = tokenizer(
              prompts[i:i+batch_size],
              return_tensors="pt",
              padding=True,
              truncation=True,
              max_length=seq_len
          )

          input_ids = inputs.input_ids.to(model.device)
          attention_mask = inputs.attention_mask.to(model.device)

          inputs_len = len(input_ids)
          num_input_toks = input_ids.shape[-1]

          # CREATE HOOKS FOR EACH LAYER
          # Key insight: Hook the specific slice of the cache tensor for this batch
          fwd_pre_hooks = [
              (
                  block_modules[layer],
                  _get_activations_pre_hook(
                      cache=activations[i:i+inputs_len, layer_idx, -num_input_toks:, :]
                  )
              )
              for layer_idx, layer in enumerate(layers)
          ]

          # EXECUTE MODEL WITH HOOKS
          with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
              model(input_ids=input_ids, attention_mask=attention_mask)

          # Store the input_ids for later token position finding
          all_input_ids[i:i+inputs_len, -num_input_toks:] = input_ids

      return all_input_ids, activations

  CRITICAL DESIGN DECISIONS:
  - Right-aligned storage: activations[..., -num_input_toks:, :] handles variable-length inputs
  - High-precision accumulation: Prevents numerical drift in large datasets
  - GPU memory management: Immediate transfer to save_device prevents OOM

  1.3 Token-Specific Position Finding

  Binary Search Implementation (utils/utils.py:319-341):
  def find_string_in_tokens(target, tokens, tokenizer) -> slice:
      assert target in tokenizer.decode(tokens), f"Target {target} not in tokens"

      # BINARY SEARCH FOR END INDEX
      end_idx_left, end_idx_right = 0, len(tokens)
      while end_idx_left != end_idx_right:
          mid = (end_idx_right + end_idx_left) // 2
          if target in tokenizer.decode(tokens[:mid]):
              end_idx_right = mid
          else:
              end_idx_left = mid + 1
      end_idx = end_idx_left

      # BINARY SEARCH FOR START INDEX
      start_idx_left, start_idx_right = 0, end_idx-1
      while start_idx_left != start_idx_right:
          mid = (start_idx_right + start_idx_left + 1) // 2
          if target in tokenizer.decode(tokens[mid:end_idx]):
              start_idx_left = mid
          else:
              start_idx_right = mid-1
      start_idx = start_idx_left

      target_slice = slice(start_idx, end_idx)
      assert target in tokenizer.decode(tokens[target_slice])
      return target_slice

  WHY BINARY SEARCH: Tokenizers can split words unexpectedly. This finds the exact token span for any substring, crucial for entity token
  identification.

  ---
  2. MEMORY-MAPPED STORAGE SYSTEM

  2.1 Efficient Large-Scale Storage

  Memory-Mapped File Creation (utils/activation_cache.py:137-140):
  # Create memory-mapped files for efficient I/O
  memmap_file_acts = np.memmap(
      f"{foldername}/acts.dat",
      dtype='float32',
      mode='w+',
      shape=(shard_size, n_layers, n_positions, d_model)
  )
  memmap_file_ids = np.memmap(
      f"{foldername}/ids.dat",
      dtype='float32',
      mode='w+',
      shape=(shard_size, seq_len)
  )

  2.2 Position-Aware Activation Slicing

  Critical Implementation (utils/activation_cache.py:166-185):
  # Filter activations based on substrings (entity names)
  if tokens_to_cache is not None:
      # Find token positions for each entity name
      slices_to_cache = [
          find_string_in_tokens(substring, input_ids[j], model_base.tokenizer)
          for j, substring in enumerate(batch_substrings)
      ]

      activations_sliced_list = []
      input_ids_list = []

      for j, s in enumerate(slices_to_cache):
          if s is None:
              continue

          # Extract activations at the END of entity tokens
          left_pos = s.stop - n_positions  # n_positions=1 means last token only

          if i == 0 and j == 0:  # Debug info for first item
              print('tokens', model_base.tokenizer.convert_ids_to_tokens(input_ids[j]))
              print('token(s) cached:', model_base.tokenizer.decode(input_ids[j, left_pos:s.stop]))
              print('left_pos', left_pos)

          # Extract: [layers, n_positions, d_model]
          activations_sliced = activations[j, :, left_pos:s.stop, :]
          activations_sliced_list.append(activations_sliced)
          input_ids_list.append(input_ids[j])

      activations = torch.stack(activations_sliced_list, dim=0)
      input_ids = torch.stack(input_ids_list, dim=0)

  KEY INSIGHT: left_pos = s.stop - n_positions extracts activations at the final token(s) of entity names, where the model has fully processed the
  entity information.

  2.3 Saving Activations to Memory-Mapped Files

  Writing to Disk (utils/activation_cache.py:196-207):
  # Store activations in memory-mapped files
  added_batch_size = activations.shape[0]
  if added_batch_size == 0:
      continue

  end_idx = min(shard_size, i + added_batch_size)
  if end_idx == shard_size:
      # If reaching end of shard, take remaining examples
      n = end_idx - i
  else:
      # Add the full batch
      n = added_batch_size

  total_n += n

  # Write to memory-mapped files (avoid upcast to float?)
  memmap_file_acts[i:end_idx] = activations[:n].float().cpu().numpy()
  memmap_file_ids[i:end_idx] = input_ids[:n].cpu().numpy()

  if end_idx == shard_size:
      print('Files loaded')
      print('total_n', total_n)
      break

  # Flush changes to disk
  memmap_file_acts.flush()
  memmap_file_ids.flush()

  YES, THEY SAVE ACTIVATIONS: The activations are saved to disk using memory-mapped files after being collected through hooks. This happens in two
  files:
  - acts.dat: Contains the actual activation tensors
  - ids.dat: Contains the corresponding input token IDs

  ---
  3. SAE LOADING AND INTEGRATION

  3.1 Multi-Source SAE Loading

  Complete SAE Loader (utils/sae_utils.py:130-156):
  def load_sae(repo_id, sae_id):
      # SAE-Lens Integration
      if repo_id == 'gemma-2b-it-res-jb':
          sae, cfg_dict, sparsity = SAE.from_pretrained(
              release=repo_id,
              sae_id="blocks.12.hook_resid_post",
              device="cuda"
          )
      elif repo_id == 'llama_scope_lxr_8x':
          sae, cfg_dict, sparsity = SAE.from_pretrained(
              release=repo_id,
              sae_id=sae_id,
              device="cuda"
          )
      # Gemma Scope SAEs from HuggingFace Hub
      else:
          path_to_params = hf_hub_download(
              repo_id=repo_id,
              filename=f"{sae_id}/params.npz",
              force_download=False,
          )
          params = np.load(path_to_params)
          pt_params = {k: torch.from_numpy(v).cuda() for k, v in params.items()}

          # Custom JumpReLU SAE implementation
          sae = JumpReLUSAE(params['W_enc'].shape[0], params['W_enc'].shape[1])
          sae.load_state_dict(pt_params)

      sae.to('cuda')
      return sae

  3.2 JumpReLU SAE Implementation

  Custom SAE Architecture (utils/sae_utils.py:102-129):
  class JumpReLUSAE(nn.Module):
      def __init__(self, d_model, d_sae):
          super().__init__()
          # All weights initialized to zeros for pre-trained loading
          self.W_enc = nn.Parameter(torch.zeros(d_model, d_sae))
          self.W_dec = nn.Parameter(torch.zeros(d_sae, d_model))
          self.threshold = nn.Parameter(torch.zeros(d_sae))
          self.b_enc = nn.Parameter(torch.zeros(d_sae))
          self.b_dec = nn.Parameter(torch.zeros(d_model))

      def encode(self, input_acts):
          pre_acts = input_acts @ self.W_enc + self.b_enc
          mask = (pre_acts > self.threshold)
          acts = mask * torch.nn.functional.relu(pre_acts)
          return acts

      def decode(self, acts):
          return acts @ self.W_dec + self.b_dec

      def forward(self, acts):
          acts = self.encode(acts)
          recon = self.decode(acts)
          return recon

  ---
  4. COMPLETE WORKFLOW IMPLEMENTATION

  4.1 Command Line Usage

  # Cache entity token activations
  python -m utils.activation_cache \
      --model_alias gemma-2-2b \
      --tokens_to_cache entity \
      --batch_size 128 \
      --entity_type_and_entity_name_format \
      --dataset wikidata

  # Cache random token activations for filtering
  python -m utils.activation_cache \
      --model_alias gemma-2-2b \
      --tokens_to_cache random \
      --batch_size 128 \
      --dataset pile

  4.2 Loading Cached Activations

  Dataset Loading (utils/activation_cache.py:290-319):
  class CachedDataset(TorchDataset):
      def __init__(self, path_name: str, layers: list, d_model: int, 
                   seq_len: int, n_positions: int = 1, 
                   shard_size: int | None = None, num_examples: int | None = None):

          acts_data_path = f"{path_name}/acts.dat"
          input_ids_path = f"{path_name}/ids.dat"

          # Load memory-mapped arrays
          acts = np.memmap(acts_data_path, dtype='float32', mode="r",
                          shape=(shard_size, len(layers), n_positions, d_model))
          input_ids = np.memmap(input_ids_path, dtype='float32', mode="r",
                               shape=(shard_size, seq_len))

          if num_examples is not None:
              acts = acts[:num_examples]
              input_ids = input_ids[:num_examples]

          self.mmap = np.array(acts)
          self.input_ids = np.array(input_ids)

      def __getitem__(self, idx):
          return (
              torch.from_numpy(np.asarray(self.mmap[idx].copy())),
              torch.from_numpy(np.asarray(self.input_ids[idx].copy())).int()
          )

  4.3 Feature Analysis Pipeline

  Complete Analysis Function (mech_interp/feature_analysis_utils.py:156-182):
  def get_per_layer_latent_scores(model_alias, tokenizer, n_layers, d_model, 
                                 sae_layers, save=False, **kwargs):
      dataset_name = kwargs['dataset_name']
      batch_size = 16
      repo_id = model_alias_to_sae_repo_id[model_alias]

      # Get dataloader for cached activations
      dataloader = get_dataloader(model_alias, kwargs['tokens_to_cache'],
                                 n_layers, d_model, dataset_name=dataset_name,
                                 batch_size=batch_size)

      # Get cached activations and labels
      acts_labels_dict = get_acts_labels_dict_(model_alias, tokenizer, dataloader,
                                             sae_layers, **kwargs)

      # Get features info per layer and save as JSON files
      get_features_layers(model_alias, acts_labels_dict, sae_layers,
                         SAE_WIDTH, repo_id, save, **kwargs)

  This guide shows the complete end-to-end process: activations are collected via hooks, saved to memory-mapped files, then loaded later for SAE
  analysis. The key insight is that they cache activations at specific token positions (like entity names) rather than storing full sequences, making
   the analysis both efficient and targeted.