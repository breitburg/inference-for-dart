You are an expert in creating elegant Dart-C++ bridges using Foreign Function Interface (FFI). Your expertise lies in crafting beautiful, maintainable APIs that adhere to Dart best practices while efficiently leveraging native C++ libraries.

## Your Core Values

1. **Design Elegance**: Create APIs that feel natural to Dart developers, with clean interfaces that embrace Dart's language features and idioms.

2. **Best Practice Adherence**: Always follow official Dart FFI guidelines, null safety principles, and modern memory management approaches.

3. **Thoughtful Architecture**: Design bridge layers that properly separate concerns, handle errors gracefully, and provide clear documentation.

4. **C++ Integration Expertise**: Demonstrate deep understanding of handling C++ from Dart, including proper memory management, type conversions, and callback patterns.

## Your Process for Any Implementation Task

1. First, thoroughly analyze the llama.cpp Python bindings and llama.h header to understand their architecture and design choices.

2. Consider the Dart developer experience - create APIs that feel natural in Dart code.

3. Design proper abstraction layers:
   - Low-level FFI bindings that directly map to C functions
   - Mid-level safe wrappers that handle memory and errors
   - High-level Dart-idiomatic APIs that expose functionality cleanly

4. Implement proper resource management with disposable patterns, finalizers, and explicit cleanup methods.

5. Add comprehensive documentation, examples, and tests.

## Technical Guidelines

- Always use the latest Dart and FFI features, do not rely on outdated patterns
- Provide synchronous APIs where appropriate, but wrap long-running operations in Dart's asynchronous patterns
- Handle errors and exceptions properly across language boundaries
- Use extension methods, named constructors, and factory patterns where they improve API clarity
- Ensure memory safety with proper allocation/deallocation patterns

When given implementation tasks, approach them with both technical precision and elegant design sensibilities, using the llama.cpp documentation as your reference framework.

### Llama.cpp Python Bindings

```python
### `llama_cpp.Llama`

High-level Python wrapper for a llama.cpp model.
    
``` class Llama:     """High-level Python wrapper for a llama.cpp model."""     __backend_initialized = False     def __init__(         self,         model_path: str,         *,         # Model Params         n_gpu_layers: int = 0,         split_mode: int = llama_cpp.LLAMA_SPLIT_MODE_LAYER,         main_gpu: int = 0,         tensor_split: Optional[List[float]] = None,         rpc_servers: Optional[str] = None,         vocab_only: bool = False,         use_mmap: bool = True,         use_mlock: bool = False,         kv_overrides: Optional[Dict[str, Union[bool, int, float, str]]] = None,         # Context Params         seed: int = llama_cpp.LLAMA_DEFAULT_SEED,         n_ctx: int = 512,         n_batch: int = 512,         n_ubatch: int = 512,         n_threads: Optional[int] = None,         n_threads_batch: Optional[int] = None,         rope_scaling_type: Optional[             int         ] = llama_cpp.LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED,         pooling_type: int = llama_cpp.LLAMA_POOLING_TYPE_UNSPECIFIED,         rope_freq_base: float = 0.0,         rope_freq_scale: float = 0.0,         yarn_ext_factor: float = -1.0,         yarn_attn_factor: float = 1.0,         yarn_beta_fast: float = 32.0,         yarn_beta_slow: float = 1.0,         yarn_orig_ctx: int = 0,         logits_all: bool = False,         embedding: bool = False,         offload_kqv: bool = True,         flash_attn: bool = False,         # Sampling Params         no_perf: bool = False,         last_n_tokens_size: int = 64,         # LoRA Params         lora_base: Optional[str] = None,         lora_scale: float = 1.0,         lora_path: Optional[str] = None,         # Backend Params         numa: Union[bool, int] = False,         # Chat Format Params         chat_format: Optional[str] = None,         chat_handler: Optional[llama_chat_format.LlamaChatCompletionHandler] = None,         # Speculative Decoding         draft_model: Optional[LlamaDraftModel] = None,         # Tokenizer Override         tokenizer: Optional[BaseLlamaTokenizer] = None,         # KV cache quantization         type_k: Optional[int] = None,         type_v: Optional[int] = None,         # Misc         spm_infill: bool = False,         verbose: bool = True,         # Extra Params         **kwargs,  # type: ignore     ):         """Load a llama.cpp model from `model_path`.         Examples:             Basic usage             >>> import llama_cpp             >>> model = llama_cpp.Llama(             ...     model_path="path/to/model",             ... )             >>> print(model("The quick brown fox jumps ", stop=["."])["choices"][0]["text"])             the lazy dog             Loading a chat model             >>> import llama_cpp             >>> model = llama_cpp.Llama(             ...     model_path="path/to/model",             ...     chat_format="llama-2",             ... )             >>> print(model.create_chat_completion(             ...     messages=[{             ...         "role": "user",             ...         "content": "what is the meaning of life?"             ...     }]             ... ))         Args:             model_path: Path to the model.             n_gpu_layers: Number of layers to offload to GPU (-ngl). If -1, all layers are offloaded.             split_mode: How to split the model across GPUs. See llama_cpp.LLAMA_SPLIT_* for options.             main_gpu: main_gpu interpretation depends on split_mode: LLAMA_SPLIT_MODE_NONE: the GPU that is used for the entire model. LLAMA_SPLIT_MODE_ROW: the GPU that is used for small tensors and intermediate results. LLAMA_SPLIT_MODE_LAYER: ignored             tensor_split: How split tensors should be distributed across GPUs. If None, the model is not split.             rpc_servers: Comma separated list of RPC servers to use for offloading             vocab_only: Only load the vocabulary no weights.             use_mmap: Use mmap if possible.             use_mlock: Force the system to keep the model in RAM.             kv_overrides: Key-value overrides for the model.             seed: RNG seed, -1 for random             n_ctx: Text context, 0 = from model             n_batch: Prompt processing maximum batch size             n_ubatch: Physical batch size             n_threads: Number of threads to use for generation             n_threads_batch: Number of threads to use for batch processing             rope_scaling_type: RoPE scaling type, from `enum llama_rope_scaling_type`. ref: https://github.com/ggerganov/llama.cpp/pull/2054             pooling_type: Pooling type, from `enum llama_pooling_type`.             rope_freq_base: RoPE base frequency, 0 = from model             rope_freq_scale: RoPE frequency scaling factor, 0 = from model             yarn_ext_factor: YaRN extrapolation mix factor, negative = from model             yarn_attn_factor: YaRN magnitude scaling factor             yarn_beta_fast: YaRN low correction dim             yarn_beta_slow: YaRN high correction dim             yarn_orig_ctx: YaRN original context size             logits_all: Return logits for all tokens, not just the last token. Must be True for completion to return logprobs.             embedding: Embedding mode only.             offload_kqv: Offload K, Q, V to GPU.             flash_attn: Use flash attention.             no_perf: Measure performance timings.             last_n_tokens_size: Maximum number of tokens to keep in the last_n_tokens deque.             lora_base: Optional path to base model, useful if using a quantized base model and you want to apply LoRA to an f16 model.             lora_path: Path to a LoRA file to apply to the model.             numa: numa policy             chat_format: String specifying the chat format to use when calling create_chat_completion.             chat_handler: Optional chat handler to use when calling create_chat_completion.             draft_model: Optional draft model to use for speculative decoding.             tokenizer: Optional tokenizer to override the default tokenizer from llama.cpp.             verbose: Print verbose output to stderr.             type_k: KV cache data type for K (default: f16)             type_v: KV cache data type for V (default: f16)             spm_infill: Use Suffix/Prefix/Middle pattern for infill (instead of Prefix/Suffix/Middle) as some models prefer this.         Raises:             ValueError: If the model path does not exist.         Returns:             A Llama instance.         """         self.verbose = verbose         self._stack = contextlib.ExitStack()         set_verbose(verbose)         if not Llama.__backend_initialized:             with suppress_stdout_stderr(disable=verbose):                 llama_cpp.llama_backend_init()             Llama.__backend_initialized = True         if isinstance(numa, bool):             self.numa = (                 llama_cpp.GGML_NUMA_STRATEGY_DISTRIBUTE                 if numa                 else llama_cpp.GGML_NUMA_STRATEGY_DISABLED             )         else:             self.numa = numa         if self.numa != llama_cpp.GGML_NUMA_STRATEGY_DISABLED:             with suppress_stdout_stderr(disable=verbose):                 llama_cpp.llama_numa_init(self.numa)         self.model_path = model_path         # Model Params         self.model_params = llama_cpp.llama_model_default_params()         self.model_params.n_gpu_layers = (             0x7FFFFFFF if n_gpu_layers == -1 else n_gpu_layers         )  # 0x7FFFFFFF is INT32 max, will be auto set to all layers         self.model_params.split_mode = split_mode         self.model_params.main_gpu = main_gpu         if rpc_servers is not None:             self.model_params.rpc_servers = rpc_servers.encode("utf-8")             self._rpc_servers = rpc_servers         else:             self._rpc_servers = None         self.tensor_split = tensor_split         self._c_tensor_split = None         if self.tensor_split is not None:             if len(self.tensor_split) > llama_cpp.LLAMA_MAX_DEVICES:                 raise ValueError(                     f"Attempt to split tensors that exceed maximum supported devices. Current LLAMA_MAX_DEVICES={llama_cpp.LLAMA_MAX_DEVICES}"                 )             # Type conversion and expand the list to the length of LLAMA_MAX_DEVICES             FloatArray = ctypes.c_float * llama_cpp.LLAMA_MAX_DEVICES             self._c_tensor_split = FloatArray(                 *tensor_split  # type: ignore             )  # keep a reference to the array so it is not gc'd             self.model_params.tensor_split = self._c_tensor_split         self.model_params.vocab_only = vocab_only         self.model_params.use_mmap = use_mmap if lora_path is None else False         self.model_params.use_mlock = use_mlock         # kv_overrides is the original python dict         self.kv_overrides = kv_overrides         if kv_overrides is not None:             # _kv_overrides_array is a ctypes.Array of llama_model_kv_override Structs             kvo_array_len = len(kv_overrides) + 1  # for sentinel element             self._kv_overrides_array = (                 llama_cpp.llama_model_kv_override * kvo_array_len             )()             for i, (k, v) in enumerate(kv_overrides.items()):                 self._kv_overrides_array[i].key = k.encode("utf-8")                 if isinstance(v, bool):                     self._kv_overrides_array[                         i                     ].tag = llama_cpp.LLAMA_KV_OVERRIDE_TYPE_BOOL                     self._kv_overrides_array[i].value.val_bool = v                 elif isinstance(v, int):                     self._kv_overrides_array[                         i                     ].tag = llama_cpp.LLAMA_KV_OVERRIDE_TYPE_INT                     self._kv_overrides_array[i].value.val_i64 = v                 elif isinstance(v, float):                     self._kv_overrides_array[                         i                     ].tag = llama_cpp.LLAMA_KV_OVERRIDE_TYPE_FLOAT                     self._kv_overrides_array[i].value.val_f64 = v                 elif isinstance(v, str):  # type: ignore                     v_bytes = v.encode("utf-8")                     if len(v_bytes) > 128:  # TODO: Make this a constant                         raise ValueError(f"Value for {k} is too long: {v}")                     v_bytes = v_bytes.ljust(128, b"\0")                     self._kv_overrides_array[                         i                     ].tag = llama_cpp.LLAMA_KV_OVERRIDE_TYPE_STR                     # copy min(v_bytes, 128) to str_value                     address = typing.cast(                         int,                         ctypes.addressof(self._kv_overrides_array[i].value)                         + llama_cpp.llama_model_kv_override_value.val_str.offset,                     )                     buffer_start = ctypes.cast(address, ctypes.POINTER(ctypes.c_char))                     ctypes.memmove(                         buffer_start,                         v_bytes,                         128,                     )                 else:                     raise ValueError(f"Unknown value type for {k}: {v}")             self._kv_overrides_array[                 -1             ].key = b"\0"  # ensure sentinel element is zeroed             self.model_params.kv_overrides = self._kv_overrides_array         self.n_batch = min(n_ctx, n_batch)  # ???         self.n_threads = n_threads or max(multiprocessing.cpu_count() // 2, 1)         self.n_threads_batch = n_threads_batch or multiprocessing.cpu_count()         # Used by the sampler         self._seed = seed or llama_cpp.LLAMA_DEFAULT_SEED         # Context Params         self.context_params = llama_cpp.llama_context_default_params()         self.context_params.n_ctx = n_ctx         self.context_params.n_batch = self.n_batch         self.context_params.n_ubatch = min(self.n_batch, n_ubatch)         self.context_params.n_threads = self.n_threads         self.context_params.n_threads_batch = self.n_threads_batch         self.context_params.rope_scaling_type = (             rope_scaling_type             if rope_scaling_type is not None             else llama_cpp.LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED         )         self.context_params.pooling_type = pooling_type         self.context_params.rope_freq_base = (             rope_freq_base if rope_freq_base != 0.0 else 0         )         self.context_params.rope_freq_scale = (             rope_freq_scale if rope_freq_scale != 0.0 else 0         )         self.context_params.yarn_ext_factor = (             yarn_ext_factor if yarn_ext_factor != 0.0 else 0         )         self.context_params.yarn_attn_factor = (             yarn_attn_factor if yarn_attn_factor != 0.0 else 0         )         self.context_params.yarn_beta_fast = (             yarn_beta_fast if yarn_beta_fast != 0.0 else 0         )         self.context_params.yarn_beta_slow = (             yarn_beta_slow if yarn_beta_slow != 0.0 else 0         )         self.context_params.yarn_orig_ctx = yarn_orig_ctx if yarn_orig_ctx != 0 else 0         self.context_params.logits_all = (             logits_all if draft_model is None else True         )  # Must be set to True for speculative decoding         self.context_params.embeddings = embedding  # TODO: Rename to embeddings         self.context_params.offload_kqv = offload_kqv         self.context_params.flash_attn = flash_attn         #  KV cache quantization         if type_k is not None:             self.context_params.type_k = type_k         if type_v is not None:             self.context_params.type_v = type_v         # Sampling Params         self.context_params.no_perf = no_perf         self.last_n_tokens_size = last_n_tokens_size         self.cache: Optional[BaseLlamaCache] = None         self.lora_base = lora_base         self.lora_scale = lora_scale         self.lora_path = lora_path         self.spm_infill = spm_infill         if not os.path.exists(model_path):             raise ValueError(f"Model path does not exist: {model_path}")         self._model = self._stack.enter_context(             contextlib.closing(                 internals.LlamaModel(                     path_model=self.model_path,                     params=self.model_params,                     verbose=self.verbose,                 )             )         )         # Override tokenizer         self.tokenizer_ = tokenizer or LlamaTokenizer(self)         # Set the default value for the context and correct the batch         if n_ctx == 0:             n_ctx = self._model.n_ctx_train()             self.n_batch = min(n_ctx, n_batch)             self.context_params.n_ctx = self._model.n_ctx_train()             self.context_params.n_batch = self.n_batch             self.context_params.n_ubatch = min(self.n_batch, n_ubatch)         self._ctx = self._stack.enter_context(             contextlib.closing(                 internals.LlamaContext(                     model=self._model,                     params=self.context_params,                     verbose=self.verbose,                 )             )         )         self._batch = self._stack.enter_context(             contextlib.closing(                 internals.LlamaBatch(                     n_tokens=self.n_batch,                     embd=0,                     n_seq_max=self.context_params.n_ctx,                     verbose=self.verbose,                 )             )         )         self._lora_adapter: Optional[llama_cpp.llama_adapter_lora_p] = None         if self.lora_path:             self._lora_adapter = llama_cpp.llama_adapter_lora_init(                 self._model.model,                 self.lora_path.encode("utf-8"),             )             if self._lora_adapter is None:                 raise RuntimeError(                     f"Failed to initialize LoRA adapter from lora path: {self.lora_path}"                 )             def free_lora_adapter():                 if self._lora_adapter is None:                     return                 llama_cpp.llama_adapter_lora_free(self._lora_adapter)                 self._lora_adapter = None             self._stack.callback(free_lora_adapter)             if llama_cpp.llama_set_adapter_lora(                 self._ctx.ctx, self._lora_adapter, self.lora_scale             ):                 raise RuntimeError(                     f"Failed to set LoRA adapter from lora path: {self.lora_path}"                 )         if self.verbose:             print(llama_cpp.llama_print_system_info().decode("utf-8"), file=sys.stderr)         self.chat_format = chat_format         self.chat_handler = chat_handler         self._chat_handlers: Dict[             str, llama_chat_format.LlamaChatCompletionHandler         ] = {}         self.draft_model = draft_model         self._n_vocab = self.n_vocab()         self._n_ctx = self.n_ctx()         self._token_nl = self.token_nl()         self._token_eos = self.token_eos()         self._candidates = internals.LlamaTokenDataArray(n_vocab=self._n_vocab)         self.n_tokens = 0         self.input_ids: npt.NDArray[np.intc] = np.ndarray((n_ctx,), dtype=np.intc)         self.scores: npt.NDArray[np.single] = np.ndarray(             (n_ctx if logits_all == True else n_batch, self._n_vocab), dtype=np.single         )         self._mirostat_mu = ctypes.c_float(             2.0 * 5.0         )  # TODO: Move this to sampling context         try:             self.metadata = self._model.metadata()         except Exception as e:             self.metadata = {}             if self.verbose:                 print(f"Failed to load metadata: {e}", file=sys.stderr)         if self.verbose:             print(f"Model metadata: {self.metadata}", file=sys.stderr)         eos_token_id = self.token_eos()         bos_token_id = self.token_bos()         eos_token = (             self._model.token_get_text(eos_token_id) if eos_token_id != -1 else ""         )         bos_token = (             self._model.token_get_text(bos_token_id) if bos_token_id != -1 else ""         )         # Unfortunately the llama.cpp API does not return metadata arrays, so we can't get template names from tokenizer.chat_templates         template_choices = dict(             (name[10:], template)             for name, template in self.metadata.items()             if name.startswith("tokenizer.chat_template.")         )         if "tokenizer.chat_template" in self.metadata:             template_choices["chat_template.default"] = self.metadata[                 "tokenizer.chat_template"             ]         if self.verbose and template_choices:             print(                 f"Available chat formats from metadata: {', '.join(template_choices.keys())}",                 file=sys.stderr,             )         for name, template in template_choices.items():             self._chat_handlers[name] = llama_chat_format.Jinja2ChatFormatter(                 template=template,                 eos_token=eos_token,                 bos_token=bos_token,                 stop_token_ids=[eos_token_id],             ).to_chat_handler()         if (             self.chat_format is None             and self.chat_handler is None             and "chat_template.default" in template_choices         ):             chat_format = llama_chat_format.guess_chat_format_from_gguf_metadata(                 self.metadata             )             if chat_format is not None:                 self.chat_format = chat_format                 if self.verbose:                     print(f"Guessed chat format: {chat_format}", file=sys.stderr)             else:                 if self.verbose:                     print(                         f"Using gguf chat template: {template_choices['chat_template.default']}",                         file=sys.stderr,                     )                     print(f"Using chat eos_token: {eos_token}", file=sys.stderr)                     print(f"Using chat bos_token: {bos_token}", file=sys.stderr)                 self.chat_format = "chat_template.default"         if self.chat_format is None and self.chat_handler is None:             self.chat_format = "llama-2"             if self.verbose:                 print(                     f"Using fallback chat format: {self.chat_format}", file=sys.stderr                 )         self._sampler = None     @property     def ctx(self) -> llama_cpp.llama_context_p:         return self._ctx.ctx     @property     def model(self) -> llama_cpp.llama_model_p:         return self._model.model     @property     def _input_ids(self) -> npt.NDArray[np.intc]:         return self.input_ids[: self.n_tokens]     @property     def _scores(self) -> npt.NDArray[np.single]:         return self.scores[: self.n_tokens, :]     @property     def eval_tokens(self) -> Deque[int]:         return deque(self.input_ids[: self.n_tokens].tolist(), maxlen=self._n_ctx)     @property     def eval_logits(self) -> Deque[List[float]]:         return deque(             self.scores[: self.n_tokens, :].tolist(),             maxlen=self._n_ctx if self.context_params.logits_all else 1,         )     def tokenize(         self, text: bytes, add_bos: bool = True, special: bool = False     ) -> List[int]:         """Tokenize a string.         Args:             text: The utf-8 encoded string to tokenize.             add_bos: Whether to add a beginning of sequence token.             special: Whether to tokenize special tokens.         Raises:             RuntimeError: If the tokenization failed.         Returns:             A list of tokens.         """         return self.tokenizer_.tokenize(text, add_bos, special)     def detokenize(         self,         tokens: List[int],         prev_tokens: Optional[List[int]] = None,         special: bool = False,     ) -> bytes:         """Detokenize a list of tokens.         Args:             tokens: The list of tokens to detokenize.             prev_tokens: The list of previous tokens. Offset mapping will be performed if provided.             special: Whether to detokenize special tokens.         Returns:             The detokenized string.         """         return self.tokenizer_.detokenize(             tokens, prev_tokens=prev_tokens, special=special         )     def set_cache(self, cache: Optional[BaseLlamaCache]):         """Set the cache.         Args:             cache: The cache to set.         """         self.cache = cache     def set_seed(self, seed: int):         """Set the random seed.         Args:             seed: The random seed.         """         self._seed = seed     def reset(self):         """Reset the model state."""         self.n_tokens = 0     def eval(self, tokens: Sequence[int]):         """Evaluate a list of tokens.         Args:             tokens: The list of tokens to evaluate.         """         self._ctx.kv_cache_seq_rm(-1, self.n_tokens, -1)         for i in range(0, len(tokens), self.n_batch):             batch = tokens[i : min(len(tokens), i + self.n_batch)]             n_past = self.n_tokens             n_tokens = len(batch)             self._batch.set_batch(                 batch=batch, n_past=n_past, logits_all=self.context_params.logits_all             )             self._ctx.decode(self._batch)             # Save tokens             self.input_ids[n_past : n_past + n_tokens] = batch             # Save logits             if self.context_params.logits_all:                 rows = n_tokens                 cols = self._n_vocab                 logits = np.ctypeslib.as_array(                     self._ctx.get_logits(), shape=(rows * cols,)                 )                 self.scores[n_past : n_past + n_tokens, :].reshape(-1)[::] = logits             else:                 # rows = 1                 # cols = self._n_vocab                 # logits = np.ctypeslib.as_array(                 #     self._ctx.get_logits(), shape=(rows * cols,)                 # )                 # self.scores[n_past + n_tokens - 1, :].reshape(-1)[::] = logits                 # NOTE: Now that sampling is done inside the sampler, logits are only needed for logprobs which requires logits_all                 pass             # Update n_tokens             self.n_tokens += n_tokens     def _init_sampler(         self,         top_k: int = 40,         top_p: float = 0.95,         min_p: float = 0.05,         typical_p: float = 1.0,         temp: float = 0.80,         repeat_penalty: float = 1.0,         frequency_penalty: float = 0.0,         presence_penalty: float = 0.0,         tfs_z: float = 1.0,         mirostat_mode: int = 0,         mirostat_eta: float = 0.1,         mirostat_tau: float = 5.0,         penalize_nl: bool = True,         logits_processor: Optional[LogitsProcessorList] = None,         grammar: Optional[LlamaGrammar] = None,     ):         sampler = internals.LlamaSampler()         if logits_processor is not None:             # Create and add a custom sampler             def apply_func(token_data_array: llama_cpp.llama_token_data_array_p):                 size = token_data_array.contents.size                 data_soa = token_data_array.contents.data                 data_soa_address = ctypes.addressof(data_soa.contents)                 # NOTE: This is probably broken                 recarray = np.recarray(                     shape=(size,),                     dtype=np.dtype(                         [("id", np.intc), ("logit", np.single), ("p", np.single)],                         align=True,                     ),                     buf=(llama_cpp.llama_token_data * size).from_address(                         data_soa_address                     ),                 )                 for logit_processor in logits_processor:                     recarray.logit[:] = logit_processor(self._input_ids, recarray.logit)             sampler.add_custom(apply_func)         sampler.add_penalties(             n_vocab=self._n_vocab,             special_eos_id=self._token_eos,             linefeed_id=self._token_nl,             penalty_last_n=self.last_n_tokens_size,             penalty_repeat=repeat_penalty,             penalty_freq=frequency_penalty,             penalty_present=presence_penalty,             penalize_nl=penalize_nl,             ignore_eos=False,         )         if grammar is not None:             sampler.add_grammar(self._model, grammar)         if temp < 0.0:             sampler.add_softmax()             sampler.add_dist(self._seed)         elif temp == 0.0:             sampler.add_greedy()         else:             if mirostat_mode == 1:                 mirostat_m = 100                 sampler.add_mirostat(                     self._n_vocab,                     self._seed,                     mirostat_tau,                     mirostat_eta,                     mirostat_m,                 )             elif mirostat_mode == 2:                 sampler.add_mirostat_v2(                     self._seed,                     mirostat_tau,                     mirostat_eta,                 )             else:                 n_probs = 0                 min_keep = max(1, n_probs)                 sampler.add_top_k(top_k)                 sampler.add_typical(typical_p, min_keep)                 sampler.add_top_p(top_p, min_keep)                 sampler.add_min_p(min_p, min_keep)                 sampler.add_temp(temp)                 sampler.add_dist(self._seed)         return sampler     def sample(         self,         top_k: int = 40,         top_p: float = 0.95,         min_p: float = 0.05,         typical_p: float = 1.0,         temp: float = 0.80,         repeat_penalty: float = 1.0,         frequency_penalty: float = 0.0,         presence_penalty: float = 0.0,         tfs_z: float = 1.0,         mirostat_mode: int = 0,         mirostat_eta: float = 0.1,         mirostat_tau: float = 5.0,         penalize_nl: bool = True,         logits_processor: Optional[LogitsProcessorList] = None,         grammar: Optional[LlamaGrammar] = None,         idx: Optional[int] = None,     ):         """Sample a token from the model.         Args:             top_k: The top-k sampling parameter.             top_p: The top-p sampling parameter.             temp: The temperature parameter.             repeat_penalty: The repeat penalty parameter.         Returns:             The sampled token.         """         assert self.n_tokens > 0         tmp_sampler = False         if self._sampler is None:             tmp_sampler = True             self._sampler = self._init_sampler(                 top_k=top_k,                 top_p=top_p,                 min_p=min_p,                 typical_p=typical_p,                 temp=temp,                 repeat_penalty=repeat_penalty,                 frequency_penalty=frequency_penalty,                 presence_penalty=presence_penalty,                 tfs_z=tfs_z,                 mirostat_mode=mirostat_mode,                 mirostat_tau=mirostat_tau,                 mirostat_eta=mirostat_eta,                 penalize_nl=penalize_nl,                 logits_processor=logits_processor,                 grammar=grammar,             )         ridx = idx - self.n_tokens if idx is not None else -1         assert self.ctx is not None         token = self._sampler.sample(self._ctx, ridx)         if tmp_sampler:             self._sampler = None         return token     def generate(         self,         tokens: Sequence[int],         top_k: int = 40,         top_p: float = 0.95,         min_p: float = 0.05,         typical_p: float = 1.0,         temp: float = 0.80,         repeat_penalty: float = 1.0,         reset: bool = True,         frequency_penalty: float = 0.0,         presence_penalty: float = 0.0,         tfs_z: float = 1.0,         mirostat_mode: int = 0,         mirostat_tau: float = 5.0,         mirostat_eta: float = 0.1,         penalize_nl: bool = True,         logits_processor: Optional[LogitsProcessorList] = None,         stopping_criteria: Optional[StoppingCriteriaList] = None,         grammar: Optional[LlamaGrammar] = None,     ) -> Generator[int, Optional[Sequence[int]], None]:         """Create a generator of tokens from a prompt.         Examples:             >>> llama = Llama("models/ggml-7b.bin")             >>> tokens = llama.tokenize(b"Hello, world!")             >>> for token in llama.generate(tokens, top_k=40, top_p=0.95, temp=1.0, repeat_penalty=1.0):             ...     print(llama.detokenize([token]))         Args:             tokens: The prompt tokens.             top_k: The top-k sampling parameter.             top_p: The top-p sampling parameter.             temp: The temperature parameter.             repeat_penalty: The repeat penalty parameter.             reset: Whether to reset the model state.         Yields:             The generated tokens.         """         # Reset mirostat sampling         self._mirostat_mu = ctypes.c_float(2.0 * mirostat_tau)         self._sampler = self._init_sampler(             top_k=top_k,             top_p=top_p,             min_p=min_p,             typical_p=typical_p,             temp=temp,             repeat_penalty=repeat_penalty,             frequency_penalty=frequency_penalty,             presence_penalty=presence_penalty,             tfs_z=tfs_z,             mirostat_mode=mirostat_mode,             mirostat_tau=mirostat_tau,             mirostat_eta=mirostat_eta,             penalize_nl=penalize_nl,             logits_processor=logits_processor,             grammar=grammar,         )         # Check for kv cache prefix match         if reset and self.n_tokens > 0:             longest_prefix = 0             for a, b in zip(self._input_ids, tokens[:-1]):                 if a == b:                     longest_prefix += 1                 else:                     break             if longest_prefix > 0:                 reset = False                 tokens = tokens[longest_prefix:]                 self.n_tokens = longest_prefix                 if self.verbose:                     print(                         f"Llama.generate: {longest_prefix} prefix-match hit, "                         f"remaining {len(tokens)} prompt tokens to eval",                         file=sys.stderr,                     )         # Reset the model state         if reset:             self.reset()         # # Reset the grammar         # if grammar is not None:         #     grammar.reset()         sample_idx = self.n_tokens + len(tokens) - 1         tokens = list(tokens)         # Eval and sample         while True:             self.eval(tokens)             while sample_idx < self.n_tokens:                 token = self.sample(                     top_k=top_k,                     top_p=top_p,                     min_p=min_p,                     typical_p=typical_p,                     temp=temp,                     repeat_penalty=repeat_penalty,                     frequency_penalty=frequency_penalty,                     presence_penalty=presence_penalty,                     tfs_z=tfs_z,                     mirostat_mode=mirostat_mode,                     mirostat_tau=mirostat_tau,                     mirostat_eta=mirostat_eta,                     logits_processor=logits_processor,                     grammar=grammar,                     penalize_nl=penalize_nl,                     idx=sample_idx,                 )                 sample_idx += 1                 if stopping_criteria is not None and stopping_criteria(                     self._input_ids[: sample_idx], self._scores[sample_idx - self.n_tokens, :]                 ):                     return                 tokens_or_none = yield token                 tokens.clear()                 tokens.append(token)                 if tokens_or_none is not None:                     tokens.extend(tokens_or_none)                 if sample_idx < self.n_tokens and token != self._input_ids[sample_idx]:                     self.n_tokens = sample_idx                     self._ctx.kv_cache_seq_rm(-1, self.n_tokens, -1)                     break             if self.draft_model is not None:                 self.input_ids[self.n_tokens : self.n_tokens + len(tokens)] = tokens                 draft_tokens = self.draft_model(                     self.input_ids[: self.n_tokens + len(tokens)]                 )                 tokens.extend(                     draft_tokens.astype(int)[                         : self._n_ctx - self.n_tokens - len(tokens)                     ]                 )     def create_embedding(         self, input: Union[str, List[str]], model: Optional[str] = None     ) -> CreateEmbeddingResponse:         """Embed a string.         Args:             input: The utf-8 encoded string to embed.         Returns:             An embedding object.         """         model_name: str = model if model is not None else self.model_path         input = input if isinstance(input, list) else [input]         # get numeric embeddings         embeds: Union[List[List[float]], List[List[List[float]]]]         total_tokens: int         embeds, total_tokens = self.embed(input, return_count=True)  # type: ignore         # convert to CreateEmbeddingResponse         data: List[Embedding] = [             {                 "object": "embedding",                 "embedding": emb,                 "index": idx,             }             for idx, emb in enumerate(embeds)         ]         return {             "object": "list",             "data": data,             "model": model_name,             "usage": {                 "prompt_tokens": total_tokens,                 "total_tokens": total_tokens,             },         }     def embed(         self,         input: Union[str, List[str]],         normalize: bool = False,         truncate: bool = True,         return_count: bool = False,     ):         """Embed a string.         Args:             input: The utf-8 encoded string to embed.         Returns:             A list of embeddings         """         n_embd = self.n_embd()         n_batch = self.n_batch         # get pooling information         pooling_type = self.pooling_type()         logits_all = pooling_type == llama_cpp.LLAMA_POOLING_TYPE_NONE         if self.context_params.embeddings is False:             raise RuntimeError(                 "Llama model must be created with embedding=True to call this method"             )         if self.verbose:             llama_cpp.llama_perf_context_reset(self._ctx.ctx)         if isinstance(input, str):             inputs = [input]         else:             inputs = input         # reset batch         self._batch.reset()         # decode and fetch embeddings         data: Union[List[List[float]], List[List[List[float]]]] = []         def decode_batch(seq_sizes: List[int]):             llama_cpp.llama_kv_cache_clear(self._ctx.ctx)             self._ctx.decode(self._batch)             self._batch.reset()             # store embeddings             if pooling_type == llama_cpp.LLAMA_POOLING_TYPE_NONE:                 pos: int = 0                 for i, size in enumerate(seq_sizes):                     ptr = llama_cpp.llama_get_embeddings(self._ctx.ctx)                     embedding: List[List[float]] = [                         ptr[pos + j * n_embd : pos + (j + 1) * n_embd]                         for j in range(size)                     ]                     if normalize:                         embedding = [                             internals.normalize_embedding(e) for e in embedding                         ]                     data.append(embedding)                     pos += size             else:                 for i in range(len(seq_sizes)):                     ptr = llama_cpp.llama_get_embeddings_seq(self._ctx.ctx, i)                     embedding: List[float] = ptr[:n_embd]                     if normalize:                         embedding = internals.normalize_embedding(embedding)                     data.append(embedding)         # init state         total_tokens = 0         s_batch = []         t_batch = 0         p_batch = 0         # accumulate batches and encode         for text in inputs:             tokens = self.tokenize(text.encode("utf-8"))             if truncate:                 tokens = tokens[:n_batch]             n_tokens = len(tokens)             total_tokens += n_tokens             # check for overrun             if n_tokens > n_batch:                 raise ValueError(                     f"Requested tokens ({n_tokens}) exceed batch size of {n_batch}"                 )             # time to eval batch             if t_batch + n_tokens > n_batch:                 decode_batch(s_batch)                 s_batch = []                 t_batch = 0                 p_batch = 0             # add to batch             self._batch.add_sequence(tokens, p_batch, logits_all)             # update batch stats             s_batch.append(n_tokens)             t_batch += n_tokens             p_batch += 1         # hanlde last batch         decode_batch(s_batch)         if self.verbose:             llama_cpp.llama_perf_context_print(self._ctx.ctx)         output = data[0] if isinstance(input, str) else data         llama_cpp.llama_kv_cache_clear(self._ctx.ctx)         self.reset()         if return_count:             return output, total_tokens         else:             return output     def _create_completion(         self,         prompt: Union[str, List[int]],         suffix: Optional[str] = None,         max_tokens: Optional[int] = 16,         temperature: float = 0.8,         top_p: float = 0.95,         min_p: float = 0.05,         typical_p: float = 1.0,         logprobs: Optional[int] = None,         echo: bool = False,         stop: Optional[Union[str, List[str]]] = [],         frequency_penalty: float = 0.0,         presence_penalty: float = 0.0,         repeat_penalty: float = 1.0,         top_k: int = 40,         stream: bool = False,         seed: Optional[int] = None,         tfs_z: float = 1.0,         mirostat_mode: int = 0,         mirostat_tau: float = 5.0,         mirostat_eta: float = 0.1,         model: Optional[str] = None,         stopping_criteria: Optional[StoppingCriteriaList] = None,         logits_processor: Optional[LogitsProcessorList] = None,         grammar: Optional[LlamaGrammar] = None,         logit_bias: Optional[Dict[int, float]] = None,     ) -> Union[         Iterator[CreateCompletionResponse], Iterator[CreateCompletionStreamResponse]     ]:         assert suffix is None or suffix.__class__ is str         completion_id: str = f"cmpl-{str(uuid.uuid4())}"         created: int = int(time.time())         bos_token_id: int = self.token_bos()         cls_token_id: int = self._model.token_cls()         sep_token_id: int = self._model.token_sep()         prefix_token_id: int = 0 # self._model.token_prefix() # TODO: Fix         middle_token_id: int = 0 # self._model.token_middle() # TODO: Fix         suffix_token_id: int = 0 # self._model.token_suffix() # TODO: Fix         add_space_prefix: bool = (             self.metadata.get("tokenizer.ggml.add_space_prefix", "true") == "true"         )         bos_tokens: List[int] = [cls_token_id if cls_token_id != -1 else bos_token_id]         eos_tokens: List[int] = [             sep_token_id if sep_token_id != -1 else self.token_eos()         ]         if (             (isinstance(prompt, list) and suffix is None)             or not self._model.add_bos_token()             or bos_tokens[:1] == [-1]         ):             bos_tokens = []         if (isinstance(prompt, list) and suffix is None) or (             not self._model.add_eos_token() and sep_token_id == -1         ):             eos_tokens = []         suffix_space_prefix: int = 0         # Tokenizer hack to remove leading space         if add_space_prefix and suffix_token_id >= 0 and suffix:             suffix = "☺" + suffix             suffix_space_prefix = 2         # If prompt is empty, initialize completion with BOS token to avoid         # detokenization including a space at the beginning of the completion         completion_tokens: List[int] = [] if len(prompt) > 0 else [bos_token_id]         # Add blank space to start of prompt to match OG llama tokenizer         prefix_tokens: List[int] = (             [prefix_token_id] if prefix_token_id >= 0 and suffix is not None else []         ) + (             (                 self.tokenize(                     prompt.encode("utf-8"),                     add_bos=False,                     special=(prefix_token_id < 0 or suffix is None),                 )                 if prompt != ""                 else []             )             if isinstance(prompt, str)             else prompt         )         suffix_tokens: List[int] = (             (                 [suffix_token_id]                 + (                     self.tokenize(suffix.encode("utf-8"), add_bos=False, special=False)[                         suffix_space_prefix:                     ]                     if suffix                     else []                 )             )             if suffix_token_id >= 0 and suffix is not None             else []         )         middle_tokens: List[int] = (             [middle_token_id] if middle_token_id >= 0 and suffix is not None else []         )         prompt_tokens: List[int] = (             bos_tokens             + (                 (suffix_tokens + prefix_tokens + middle_tokens)                 if self.spm_infill                 else (prefix_tokens + suffix_tokens + middle_tokens)             )             + eos_tokens         )         text: bytes = b""         returned_tokens: int = 0         stop = (             stop if isinstance(stop, list) else [stop] if isinstance(stop, str) else []         )         model_name: str = model if model is not None else self.model_path         if prompt_tokens[:2] == [self.token_bos()] * 2:             warnings.warn(                 f'Detected duplicate leading "{self._model.token_get_text(self.token_bos())}" in prompt, this will likely reduce response quality, consider removing it...',                 RuntimeWarning,             )         # NOTE: This likely doesn't work correctly for the first token in the prompt         # because of the extra space added to the start of the prompt_tokens         if logit_bias is not None:             logit_bias_map = {int(k): float(v) for k, v in logit_bias.items()}             def logit_bias_processor(                 input_ids: npt.NDArray[np.intc],                 scores: npt.NDArray[np.single],             ) -> npt.NDArray[np.single]:                 new_scores = np.copy(                     scores                 )  # Does it make sense to copy the whole array or can we just overwrite the original one?                 for input_id, score in logit_bias_map.items():                     new_scores[input_id] = score + scores[input_id]                 return new_scores             _logit_bias_processor = LogitsProcessorList([logit_bias_processor])             if logits_processor is None:                 logits_processor = _logit_bias_processor             else:                 logits_processor = logits_processor.extend(_logit_bias_processor)         if self.verbose:             self._ctx.reset_timings()         if len(prompt_tokens) >= self._n_ctx:             raise ValueError(                 f"Requested tokens ({len(prompt_tokens)}) exceed context window of {llama_cpp.llama_n_ctx(self.ctx)}"             )         if max_tokens is None or max_tokens <= 0:             # Unlimited, depending on n_ctx.             max_tokens = self._n_ctx - len(prompt_tokens)         # Truncate max_tokens if requested tokens would exceed the context window         max_tokens = (             max_tokens             if max_tokens + len(prompt_tokens) < self._n_ctx             else (self._n_ctx - len(prompt_tokens))         )         if stop != []:             stop_sequences = [s.encode("utf-8") for s in stop]         else:             stop_sequences = []         if logprobs is not None and self.context_params.logits_all is False:             raise ValueError(                 "logprobs is not supported for models created with logits_all=False"             )         if self.cache:             try:                 cache_item = self.cache[prompt_tokens]                 cache_prefix_len = Llama.longest_token_prefix(                     cache_item.input_ids.tolist(), prompt_tokens                 )                 eval_prefix_len = Llama.longest_token_prefix(                     self._input_ids.tolist(), prompt_tokens                 )                 if cache_prefix_len > eval_prefix_len:                     self.load_state(cache_item)                     if self.verbose:                         print("Llama._create_completion: cache hit", file=sys.stderr)             except KeyError:                 if self.verbose:                     print("Llama._create_completion: cache miss", file=sys.stderr)         if seed is not None:             self.set_seed(seed)         else:             self.set_seed(random.Random(self._seed).randint(0, 2 ** 32))         finish_reason = "length"         multibyte_fix = 0         for token in self.generate(             prompt_tokens,             top_k=top_k,             top_p=top_p,             min_p=min_p,             typical_p=typical_p,             temp=temperature,             tfs_z=tfs_z,             mirostat_mode=mirostat_mode,             mirostat_tau=mirostat_tau,             mirostat_eta=mirostat_eta,             frequency_penalty=frequency_penalty,             presence_penalty=presence_penalty,             repeat_penalty=repeat_penalty,             stopping_criteria=stopping_criteria,             logits_processor=logits_processor,             grammar=grammar,         ):             if llama_cpp.llama_token_is_eog(self._model.vocab, token):                 text = self.detokenize(completion_tokens, prev_tokens=prompt_tokens)                 finish_reason = "stop"                 break             completion_tokens.append(token)             all_text = self.detokenize(completion_tokens, prev_tokens=prompt_tokens)             # Contains multi-byte UTF8             for k, char in enumerate(all_text[-3:]):                 k = 3 - k                 for num, pattern in [(2, 192), (3, 224), (4, 240)]:                     # Bitwise AND check                     if num > k and pattern & char == pattern:                         multibyte_fix = num - k             # Stop incomplete bytes from passing             if multibyte_fix > 0:                 multibyte_fix -= 1                 continue             any_stop = [s for s in stop_sequences if s in all_text]             if len(any_stop) > 0:                 first_stop = any_stop[0]                 text = all_text[: all_text.index(first_stop)]                 finish_reason = "stop"                 break             if stream:                 remaining_tokens = completion_tokens[returned_tokens:]                 remaining_text = self.detokenize(                     remaining_tokens,                     prev_tokens=prompt_tokens + completion_tokens[:returned_tokens],                 )                 remaining_length = len(remaining_text)                 # We want to avoid yielding any characters from                 # the generated text if they are part of a stop                 # sequence.                 first_stop_position = 0                 for s in stop_sequences:                     for i in range(min(len(s), remaining_length), 0, -1):                         if remaining_text.endswith(s[:i]):                             if i > first_stop_position:                                 first_stop_position = i                             break                 token_end_position = 0                 if logprobs is not None:                     # not sure how to handle this branch when dealing                     # with CJK output, so keep it unchanged                     for token in remaining_tokens:                         if token == bos_token_id:                             continue                         token_end_position += len(                             self.detokenize(                                 [token],                                 prev_tokens=prompt_tokens                                 + completion_tokens[:returned_tokens],                             )                         )                         # Check if stop sequence is in the token                         if token_end_position > (                             remaining_length - first_stop_position                         ):                             break                         token_str = self.detokenize(                             [token],                             prev_tokens=prompt_tokens                             + completion_tokens[:returned_tokens],                         ).decode("utf-8", errors="ignore")                         text_offset = len(prompt) + len(                             self.detokenize(                                 completion_tokens[:returned_tokens],                                 prev_tokens=prompt_tokens                                 + completion_tokens[:returned_tokens],                             ).decode("utf-8", errors="ignore")                         )                         token_offset = len(prompt_tokens) + returned_tokens                         logits = self._scores[token_offset - 1, :]                         current_logprobs = Llama.logits_to_logprobs(logits).tolist()                         sorted_logprobs = list(                             sorted(                                 zip(current_logprobs, range(len(current_logprobs))),                                 reverse=True,                             )                         )                         top_logprob = {                             self.detokenize([i]).decode(                                 "utf-8", errors="ignore"                             ): logprob                             for logprob, i in sorted_logprobs[:logprobs]                         }                         top_logprob.update({token_str: current_logprobs[int(token)]})                         logprobs_or_none = {                             "tokens": [                                 self.detokenize(                                     [token],                                     prev_tokens=prompt_tokens                                     + completion_tokens[:returned_tokens],                                 ).decode("utf-8", errors="ignore")                             ],                             "text_offset": [text_offset],                             "token_logprobs": [current_logprobs[int(token)]],                             "top_logprobs": [top_logprob],                         }                         returned_tokens += 1                         yield {                             "id": completion_id,                             "object": "text_completion",                             "created": created,                             "model": model_name,                             "choices": [                                 {                                     "text": self.detokenize(                                         [token],                                         prev_tokens=prompt_tokens                                         + completion_tokens[:returned_tokens],                                     ).decode("utf-8", errors="ignore"),                                     "index": 0,                                     "logprobs": logprobs_or_none,                                     "finish_reason": None,                                 }                             ],                         }                 else:                     while len(remaining_tokens) > 0:                         decode_success = False                         for i in range(1, len(remaining_tokens) + 1):                             try:                                 bs = self.detokenize(                                     remaining_tokens[:i],                                     prev_tokens=prompt_tokens                                     + completion_tokens[:returned_tokens],                                 )                                 ts = bs.decode("utf-8")                                 decode_success = True                                 break                             except UnicodeError:                                 pass                         else:                             break                         if not decode_success:                             # all remaining tokens cannot be decoded to a UTF-8 character                             break                         token_end_position += len(bs)                         if token_end_position > (                             remaining_length - first_stop_position                         ):                             break                         remaining_tokens = remaining_tokens[i:]                         returned_tokens += i                         yield {                             "id": completion_id,                             "object": "text_completion",                             "created": created,                             "model": model_name,                             "choices": [                                 {                                     "text": ts,                                     "index": 0,                                     "logprobs": None,                                     "finish_reason": None,                                 }                             ],                         }             if len(completion_tokens) >= max_tokens:                 text = self.detokenize(completion_tokens, prev_tokens=prompt_tokens)                 finish_reason = "length"                 break         if stopping_criteria is not None and stopping_criteria(             self._input_ids, self._scores[-1, :]         ):             text = self.detokenize(completion_tokens, prev_tokens=prompt_tokens)             finish_reason = "stop"         if self.verbose:             self._ctx.print_timings()         if stream:             remaining_tokens = completion_tokens[returned_tokens:]             remaining_text = self.detokenize(                 remaining_tokens,                 prev_tokens=prompt_tokens + completion_tokens[:returned_tokens],             )             any_stop = [s for s in stop_sequences if s in remaining_text]             if len(any_stop) > 0:                 end = min(remaining_text.index(stop) for stop in any_stop)             else:                 end = len(remaining_text)             token_end_position = 0             for token in remaining_tokens:                 token_end_position += len(                     self.detokenize(                         [token],                         prev_tokens=prompt_tokens + completion_tokens[:returned_tokens],                     )                 )                 logprobs_or_none: Optional[CompletionLogprobs] = None                 if logprobs is not None:                     if token == bos_token_id:                         continue                     token_str = self.detokenize([token]).decode(                         "utf-8", errors="ignore"                     )                     text_offset = len(prompt) + len(                         self.detokenize(                             completion_tokens[:returned_tokens],                             prev_tokens=prompt_tokens                             + completion_tokens[:returned_tokens],                         )                     )                     token_offset = len(prompt_tokens) + returned_tokens - 1                     logits = self._scores[token_offset, :]                     current_logprobs = Llama.logits_to_logprobs(logits).tolist()                     sorted_logprobs = list(                         sorted(                             zip(current_logprobs, range(len(current_logprobs))),                             reverse=True,                         )                     )                     top_logprob = {                         self.detokenize([i]).decode("utf-8", errors="ignore"): logprob                         for logprob, i in sorted_logprobs[:logprobs]                     }                     top_logprob.update({token_str: current_logprobs[int(token)]})                     logprobs_or_none = {                         "tokens": [                             self.detokenize([token]).decode("utf-8", errors="ignore")                         ],                         "text_offset": [text_offset],                         "token_logprobs": [current_logprobs[int(token)]],                         "top_logprobs": [top_logprob],                     }                 if token_end_position >= end:                     last_text = self.detokenize([token])                     if token_end_position == end - 1:                         break                     returned_tokens += 1                     yield {                         "id": completion_id,                         "object": "text_completion",                         "created": created,                         "model": model_name,                         "choices": [                             {                                 "text": last_text[                                     : len(last_text) - (token_end_position - end)                                 ].decode("utf-8", errors="ignore"),                                 "index": 0,                                 "logprobs": logprobs_or_none,                                 "finish_reason": None,                             }                         ],                     }                     break                 returned_tokens += 1                 yield {                     "id": completion_id,                     "object": "text_completion",                     "created": created,                     "model": model_name,                     "choices": [                         {                             "text": self.detokenize([token]).decode(                                 "utf-8", errors="ignore"                             ),                             "index": 0,                             "logprobs": logprobs_or_none,                             "finish_reason": None,                         }                     ],                 }             yield {                 "id": completion_id,                 "object": "text_completion",                 "created": created,                 "model": model_name,                 "choices": [                     {                         "text": "",                         "index": 0,                         "logprobs": None,                         "finish_reason": finish_reason,                     }                 ],             }             if self.cache:                 if self.verbose:                     print("Llama._create_completion: cache save", file=sys.stderr)                 self.cache[prompt_tokens + completion_tokens] = self.save_state()                 if self.verbose:                     print("Llama._create_completion: cache saved", file=sys.stderr)             return         if self.cache:             if self.verbose:                 print("Llama._create_completion: cache save", file=sys.stderr)             self.cache[prompt_tokens + completion_tokens] = self.save_state()         text_str = text.decode("utf-8", errors="ignore")         if echo:             text_str = prompt + text_str         if suffix_token_id < 0 and suffix is not None:             text_str = text_str + suffix         logprobs_or_none: Optional[CompletionLogprobs] = None         if logprobs is not None:             text_offset = 0 if echo else len(prompt)             token_offset = 0 if echo else len(prompt_tokens[1:])             text_offsets: List[int] = []             token_logprobs: List[Optional[float]] = []             tokens: List[str] = []             top_logprobs: List[Optional[Dict[str, float]]] = []             if echo:                 # Remove leading BOS token if exists                 all_tokens = (                     prompt_tokens[1 if prompt_tokens[0] == self.token_bos() else 0 :]                     + completion_tokens                 )             else:                 all_tokens = completion_tokens             all_token_strs = [                 self.detokenize([token], prev_tokens=all_tokens[:i]).decode(                     "utf-8", errors="ignore"                 )                 for i, token in enumerate(all_tokens)             ]             all_logprobs = Llama.logits_to_logprobs(self._scores)[token_offset:]             # TODO: may be able to change this loop to use np.take_along_dim             for idx, (token, token_str, logprobs_token) in enumerate(                 zip(all_tokens, all_token_strs, all_logprobs)             ):                 if token == bos_token_id:                     continue                 text_offsets.append(                     text_offset                     + len(                         self.detokenize(all_tokens[:idx]).decode(                             "utf-8", errors="ignore"                         )                     )                 )                 tokens.append(token_str)                 sorted_logprobs = list(                     sorted(                         zip(logprobs_token, range(len(logprobs_token))), reverse=True                     )                 )                 token_logprobs.append(logprobs_token[int(token)])                 top_logprob: Optional[Dict[str, float]] = {                     self.detokenize([i], prev_tokens=all_tokens[:idx]).decode(                         "utf-8", errors="ignore"                     ): logprob                     for logprob, i in sorted_logprobs[:logprobs]                 }                 top_logprob.update({token_str: logprobs_token[int(token)]})                 top_logprobs.append(top_logprob)             # Weird idosincracy of the OpenAI API where             # token_logprobs and top_logprobs are null for             # the first token.             if echo and len(all_tokens) > 0:                 token_logprobs[0] = None                 top_logprobs[0] = None             logprobs_or_none = {                 "tokens": tokens,                 "text_offset": text_offsets,                 "token_logprobs": token_logprobs,                 "top_logprobs": top_logprobs,             }         yield {             "id": completion_id,             "object": "text_completion",             "created": created,             "model": model_name,             "choices": [                 {                     "text": text_str,                     "index": 0,                     "logprobs": logprobs_or_none,                     "finish_reason": finish_reason,                 }             ],             "usage": {                 "prompt_tokens": len(prompt_tokens),                 "completion_tokens": len(completion_tokens),                 "total_tokens": len(prompt_tokens) + len(completion_tokens),             },         }     def create_completion(         self,         prompt: Union[str, List[int]],         suffix: Optional[str] = None,         max_tokens: Optional[int] = 16,         temperature: float = 0.8,         top_p: float = 0.95,         min_p: float = 0.05,         typical_p: float = 1.0,         logprobs: Optional[int] = None,         echo: bool = False,         stop: Optional[Union[str, List[str]]] = [],         frequency_penalty: float = 0.0,         presence_penalty: float = 0.0,         repeat_penalty: float = 1.0,         top_k: int = 40,         stream: bool = False,         seed: Optional[int] = None,         tfs_z: float = 1.0,         mirostat_mode: int = 0,         mirostat_tau: float = 5.0,         mirostat_eta: float = 0.1,         model: Optional[str] = None,         stopping_criteria: Optional[StoppingCriteriaList] = None,         logits_processor: Optional[LogitsProcessorList] = None,         grammar: Optional[LlamaGrammar] = None,         logit_bias: Optional[Dict[int, float]] = None,     ) -> Union[CreateCompletionResponse, Iterator[CreateCompletionStreamResponse]]:         """Generate text from a prompt.         Args:             prompt: The prompt to generate text from.             suffix: A suffix to append to the generated text. If None, no suffix is appended.             max_tokens: The maximum number of tokens to generate. If max_tokens <= 0 or None, the maximum number of tokens to generate is unlimited and depends on n_ctx.             temperature: The temperature to use for sampling.             top_p: The top-p value to use for nucleus sampling. Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751             min_p: The min-p value to use for minimum p sampling. Minimum P sampling as described in https://github.com/ggerganov/llama.cpp/pull/3841             typical_p: The typical-p value to use for sampling. Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666.             logprobs: The number of logprobs to return. If None, no logprobs are returned.             echo: Whether to echo the prompt.             stop: A list of strings to stop generation when encountered.             frequency_penalty: The penalty to apply to tokens based on their frequency in the prompt.             presence_penalty: The penalty to apply to tokens based on their presence in the prompt.             repeat_penalty: The penalty to apply to repeated tokens.             top_k: The top-k value to use for sampling. Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751             stream: Whether to stream the results.             seed: The seed to use for sampling.             tfs_z: The tail-free sampling parameter. Tail Free Sampling described in https://www.trentonbricken.com/Tail-Free-Sampling/.             mirostat_mode: The mirostat sampling mode.             mirostat_tau: The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.             mirostat_eta: The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.             model: The name to use for the model in the completion object.             stopping_criteria: A list of stopping criteria to use.             logits_processor: A list of logits processors to use.             grammar: A grammar to use for constrained sampling.             logit_bias: A logit bias to use.         Raises:             ValueError: If the requested tokens exceed the context window.             RuntimeError: If the prompt fails to tokenize or the model fails to evaluate the prompt.         Returns:             Response object containing the generated text.         """         completion_or_chunks = self._create_completion(             prompt=prompt,             suffix=suffix,             max_tokens=-1 if max_tokens is None else max_tokens,             temperature=temperature,             top_p=top_p,             min_p=min_p,             typical_p=typical_p,             logprobs=logprobs,             echo=echo,             stop=stop,             frequency_penalty=frequency_penalty,             presence_penalty=presence_penalty,             repeat_penalty=repeat_penalty,             top_k=top_k,             stream=stream,             seed=seed,             tfs_z=tfs_z,             mirostat_mode=mirostat_mode,             mirostat_tau=mirostat_tau,             mirostat_eta=mirostat_eta,             model=model,             stopping_criteria=stopping_criteria,             logits_processor=logits_processor,             grammar=grammar,             logit_bias=logit_bias,         )         if stream:             chunks: Iterator[CreateCompletionStreamResponse] = completion_or_chunks             return chunks         completion: Completion = next(completion_or_chunks)  # type: ignore         return completion     def __call__(         self,         prompt: str,         suffix: Optional[str] = None,         max_tokens: Optional[int] = 16,         temperature: float = 0.8,         top_p: float = 0.95,         min_p: float = 0.05,         typical_p: float = 1.0,         logprobs: Optional[int] = None,         echo: bool = False,         stop: Optional[Union[str, List[str]]] = [],         frequency_penalty: float = 0.0,         presence_penalty: float = 0.0,         repeat_penalty: float = 1.0,         top_k: int = 40,         stream: bool = False,         seed: Optional[int] = None,         tfs_z: float = 1.0,         mirostat_mode: int = 0,         mirostat_tau: float = 5.0,         mirostat_eta: float = 0.1,         model: Optional[str] = None,         stopping_criteria: Optional[StoppingCriteriaList] = None,         logits_processor: Optional[LogitsProcessorList] = None,         grammar: Optional[LlamaGrammar] = None,         logit_bias: Optional[Dict[int, float]] = None,     ) -> Union[CreateCompletionResponse, Iterator[CreateCompletionStreamResponse]]:         """Generate text from a prompt.         Args:             prompt: The prompt to generate text from.             suffix: A suffix to append to the generated text. If None, no suffix is appended.             max_tokens: The maximum number of tokens to generate. If max_tokens <= 0 or None, the maximum number of tokens to generate is unlimited and depends on n_ctx.             temperature: The temperature to use for sampling.             top_p: The top-p value to use for nucleus sampling. Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751             min_p: The min-p value to use for minimum p sampling. Minimum P sampling as described in https://github.com/ggerganov/llama.cpp/pull/3841             typical_p: The typical-p value to use for sampling. Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666.             logprobs: The number of logprobs to return. If None, no logprobs are returned.             echo: Whether to echo the prompt.             stop: A list of strings to stop generation when encountered.             frequency_penalty: The penalty to apply to tokens based on their frequency in the prompt.             presence_penalty: The penalty to apply to tokens based on their presence in the prompt.             repeat_penalty: The penalty to apply to repeated tokens.             top_k: The top-k value to use for sampling. Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751             stream: Whether to stream the results.             seed: The seed to use for sampling.             tfs_z: The tail-free sampling parameter. Tail Free Sampling described in https://www.trentonbricken.com/Tail-Free-Sampling/.             mirostat_mode: The mirostat sampling mode.             mirostat_tau: The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.             mirostat_eta: The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.             model: The name to use for the model in the completion object.             stopping_criteria: A list of stopping criteria to use.             logits_processor: A list of logits processors to use.             grammar: A grammar to use for constrained sampling.             logit_bias: A logit bias to use.         Raises:             ValueError: If the requested tokens exceed the context window.             RuntimeError: If the prompt fails to tokenize or the model fails to evaluate the prompt.         Returns:             Response object containing the generated text.         """         return self.create_completion(             prompt=prompt,             suffix=suffix,             max_tokens=max_tokens,             temperature=temperature,             top_p=top_p,             min_p=min_p,             typical_p=typical_p,             logprobs=logprobs,             echo=echo,             stop=stop,             frequency_penalty=frequency_penalty,             presence_penalty=presence_penalty,             repeat_penalty=repeat_penalty,             top_k=top_k,             stream=stream,             seed=seed,             tfs_z=tfs_z,             mirostat_mode=mirostat_mode,             mirostat_tau=mirostat_tau,             mirostat_eta=mirostat_eta,             model=model,             stopping_criteria=stopping_criteria,             logits_processor=logits_processor,             grammar=grammar,             logit_bias=logit_bias,         )     def create_chat_completion(         self,         messages: List[ChatCompletionRequestMessage],         functions: Optional[List[ChatCompletionFunction]] = None,         function_call: Optional[ChatCompletionRequestFunctionCall] = None,         tools: Optional[List[ChatCompletionTool]] = None,         tool_choice: Optional[ChatCompletionToolChoiceOption] = None,         temperature: float = 0.2,         top_p: float = 0.95,         top_k: int = 40,         min_p: float = 0.05,         typical_p: float = 1.0,         stream: bool = False,         stop: Optional[Union[str, List[str]]] = [],         seed: Optional[int] = None,         response_format: Optional[ChatCompletionRequestResponseFormat] = None,         max_tokens: Optional[int] = None,         presence_penalty: float = 0.0,         frequency_penalty: float = 0.0,         repeat_penalty: float = 1.0,         tfs_z: float = 1.0,         mirostat_mode: int = 0,         mirostat_tau: float = 5.0,         mirostat_eta: float = 0.1,         model: Optional[str] = None,         logits_processor: Optional[LogitsProcessorList] = None,         grammar: Optional[LlamaGrammar] = None,         logit_bias: Optional[Dict[int, float]] = None,         logprobs: Optional[bool] = None,         top_logprobs: Optional[int] = None,     ) -> Union[         CreateChatCompletionResponse, Iterator[CreateChatCompletionStreamResponse]     ]:         """Generate a chat completion from a list of messages.         Args:             messages: A list of messages to generate a response for.             functions: A list of functions to use for the chat completion.             function_call: A function call to use for the chat completion.             tools: A list of tools to use for the chat completion.             tool_choice: A tool choice to use for the chat completion.             temperature: The temperature to use for sampling.             top_p: The top-p value to use for nucleus sampling. Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751             top_k: The top-k value to use for sampling. Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751             min_p: The min-p value to use for minimum p sampling. Minimum P sampling as described in https://github.com/ggerganov/llama.cpp/pull/3841             typical_p: The typical-p value to use for sampling. Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666.             stream: Whether to stream the results.             stop: A list of strings to stop generation when encountered.             seed: The seed to use for sampling.             response_format: The response format to use for the chat completion. Use { "type": "json_object" } to contstrain output to only valid json.             max_tokens: The maximum number of tokens to generate. If max_tokens <= 0 or None, the maximum number of tokens to generate is unlimited and depends on n_ctx.             presence_penalty: The penalty to apply to tokens based on their presence in the prompt.             frequency_penalty: The penalty to apply to tokens based on their frequency in the prompt.             repeat_penalty: The penalty to apply to repeated tokens.             tfs_z: The tail-free sampling parameter.             mirostat_mode: The mirostat sampling mode.             mirostat_tau: The mirostat sampling tau parameter.             mirostat_eta: The mirostat sampling eta parameter.             model: The name to use for the model in the completion object.             logits_processor: A list of logits processors to use.             grammar: A grammar to use.             logit_bias: A logit bias to use.         Returns:             Generated chat completion or a stream of chat completion chunks.         """         handler = (             self.chat_handler             or self._chat_handlers.get(self.chat_format)             or llama_chat_format.get_chat_completion_handler(self.chat_format)         )         return handler(             llama=self,             messages=messages,             functions=functions,             function_call=function_call,             tools=tools,             tool_choice=tool_choice,             temperature=temperature,             top_p=top_p,             top_k=top_k,             min_p=min_p,             typical_p=typical_p,             logprobs=logprobs,             top_logprobs=top_logprobs,             stream=stream,             stop=stop,             seed=seed,             response_format=response_format,             max_tokens=max_tokens,             presence_penalty=presence_penalty,             frequency_penalty=frequency_penalty,             repeat_penalty=repeat_penalty,             tfs_z=tfs_z,             mirostat_mode=mirostat_mode,             mirostat_tau=mirostat_tau,             mirostat_eta=mirostat_eta,             model=model,             logits_processor=logits_processor,             grammar=grammar,             logit_bias=logit_bias,         )     def create_chat_completion_openai_v1(         self,         *args: Any,         **kwargs: Any,     ):         """Generate a chat completion with return type based on the the OpenAI v1 API.         OpenAI python package is required to use this method.         You can install it with `pip install openai`.         Args:             *args: Positional arguments to pass to create_chat_completion.             **kwargs: Keyword arguments to pass to create_chat_completion.         Returns:             Generated chat completion or a stream of chat completion chunks.         """         try:             from openai.types.chat import ChatCompletion, ChatCompletionChunk             stream = kwargs.get("stream", False)  # type: ignore             assert isinstance(stream, bool)             if stream:                 return (ChatCompletionChunk(**chunk) for chunk in self.create_chat_completion(*args, **kwargs))  # type: ignore             else:                 return ChatCompletion(**self.create_chat_completion(*args, **kwargs))  # type: ignore         except ImportError:             raise ImportError(                 "To use create_chat_completion_openai_v1, you must install the openai package."                 "You can install it with `pip install openai`."             )     def __getstate__(self):         return dict(             model_path=self.model_path,             # Model Params             n_gpu_layers=self.model_params.n_gpu_layers,             split_mode=self.model_params.split_mode,             main_gpu=self.model_params.main_gpu,             tensor_split=self.tensor_split,             vocab_only=self.model_params.vocab_only,             use_mmap=self.model_params.use_mmap,             use_mlock=self.model_params.use_mlock,             kv_overrides=self.kv_overrides,             # Context Params             seed=self._seed,             n_ctx=self.context_params.n_ctx,             n_batch=self.n_batch,             n_ubatch=self.context_params.n_ubatch,             n_threads=self.context_params.n_threads,             n_threads_batch=self.context_params.n_threads_batch,             rope_scaling_type=self.context_params.rope_scaling_type,             pooling_type=self.context_params.pooling_type,             rope_freq_base=self.context_params.rope_freq_base,             rope_freq_scale=self.context_params.rope_freq_scale,             yarn_ext_factor=self.context_params.yarn_ext_factor,             yarn_attn_factor=self.context_params.yarn_attn_factor,             yarn_beta_fast=self.context_params.yarn_beta_fast,             yarn_beta_slow=self.context_params.yarn_beta_slow,             yarn_orig_ctx=self.context_params.yarn_orig_ctx,             logits_all=self.context_params.logits_all,             embedding=self.context_params.embeddings,             offload_kqv=self.context_params.offload_kqv,             flash_attn=self.context_params.flash_attn,             # Sampling Params             no_perf=self.context_params.no_perf,             last_n_tokens_size=self.last_n_tokens_size,             # LoRA Params             lora_base=self.lora_base,             lora_scale=self.lora_scale,             lora_path=self.lora_path,             # Backend Params             numa=self.numa,             # Chat Format Params             chat_format=self.chat_format,             chat_handler=self.chat_handler,             # Speculative Decidng             draft_model=self.draft_model,             # KV cache quantization             type_k=self.context_params.type_k,             type_v=self.context_params.type_v,             # Misc             spm_infill=self.spm_infill,             verbose=self.verbose,         )     def __setstate__(self, state):         self.__init__(**state)     def save_state(self) -> LlamaState:         if self.verbose:             print("Llama.save_state: saving llama state", file=sys.stderr)         state_size = llama_cpp.llama_get_state_size(self._ctx.ctx)         if self.verbose:             print(f"Llama.save_state: got state size: {state_size}", file=sys.stderr)         llama_state = (ctypes.c_uint8 * int(state_size))()         if self.verbose:             print("Llama.save_state: allocated state", file=sys.stderr)         n_bytes = llama_cpp.llama_copy_state_data(self._ctx.ctx, llama_state)         if self.verbose:             print(f"Llama.save_state: copied llama state: {n_bytes}", file=sys.stderr)         if int(n_bytes) > int(state_size):             raise RuntimeError("Failed to copy llama state data")         llama_state_compact = (ctypes.c_uint8 * int(n_bytes))()         llama_cpp.ctypes.memmove(llama_state_compact, llama_state, int(n_bytes))         if self.verbose:             print(                 f"Llama.save_state: saving {n_bytes} bytes of llama state",                 file=sys.stderr,             )         return LlamaState(             scores=self._scores.copy(),             input_ids=self.input_ids.copy(),             n_tokens=self.n_tokens,             llama_state=bytes(llama_state_compact),             llama_state_size=n_bytes,             seed=self._seed,         )     def load_state(self, state: LlamaState) -> None:         # Only filling in up to `n_tokens` and then zero-ing out the rest         self.scores[: state.n_tokens, :] = state.scores.copy()         rest = self.scores[state.n_tokens :, :]         rest[rest > 0] = 0.0         self.input_ids = state.input_ids.copy()         self.n_tokens = state.n_tokens         self._seed = state.seed         state_size = state.llama_state_size         LLamaStateArrayType = ctypes.c_uint8 * state_size         llama_state = LLamaStateArrayType.from_buffer_copy(state.llama_state)         if llama_cpp.llama_set_state_data(self._ctx.ctx, llama_state) != state_size:             raise RuntimeError("Failed to set llama state data")     def n_ctx(self) -> int:         """Return the context window size."""         return self._ctx.n_ctx()     def n_embd(self) -> int:         """Return the embedding size."""         return self._model.n_embd()     def n_vocab(self) -> int:         """Return the vocabulary size."""         return self._model.n_vocab()     def tokenizer(self) -> LlamaTokenizer:         """Return the llama tokenizer for this model."""         return LlamaTokenizer(self)     def token_eos(self) -> int:         """Return the end-of-sequence token."""         return self._model.token_eos()     def token_bos(self) -> int:         """Return the beginning-of-sequence token."""         return self._model.token_bos()     def token_nl(self) -> int:         """Return the newline token."""         return self._model.token_nl()     def pooling_type(self) -> str:         """Return the pooling type."""         return self._ctx.pooling_type()     def close(self) -> None:         """Explicitly free the model from memory."""         self._stack.close()     def __del__(self) -> None:         self.close()     @staticmethod     def logits_to_logprobs(         logits: Union[npt.NDArray[np.single], List], axis: int = -1     ) -> npt.NDArray[np.single]:         # https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.log_softmax.html         logits_maxs: np.ndarray = np.amax(logits, axis=axis, keepdims=True)         if logits_maxs.ndim > 0:             logits_maxs[~np.isfinite(logits_maxs)] = 0         elif not np.isfinite(logits_maxs):             logits_maxs = 0         subtract_maxs = np.subtract(logits, logits_maxs, dtype=np.single)         exp = np.exp(subtract_maxs)         # Suppress warnings about log of zero         with np.errstate(divide="ignore"):             summed = np.sum(exp, axis=axis, keepdims=True)             out = np.log(summed)         return subtract_maxs - out     @staticmethod     def longest_token_prefix(a: Sequence[int], b: Sequence[int]):         longest_prefix = 0         for _a, _b in zip(a, b):             if _a == _b:                 longest_prefix += 1             else:                 break         return longest_prefix     @classmethod     def from_pretrained(         cls,         repo_id: str,         filename: Optional[str],         additional_files: Optional[List] = None,         local_dir: Optional[Union[str, os.PathLike[str]]] = None,         local_dir_use_symlinks: Union[bool, Literal["auto"]] = "auto",         cache_dir: Optional[Union[str, os.PathLike[str]]] = None,         **kwargs: Any,     ) -> "Llama":         """Create a Llama model from a pretrained model name or path.         This method requires the huggingface-hub package.         You can install it with `pip install huggingface-hub`.         Args:             repo_id: The model repo id.             filename: A filename or glob pattern to match the model file in the repo.             additional_files: A list of filenames or glob patterns to match additional model files in the repo.             local_dir: The local directory to save the model to.             local_dir_use_symlinks: Whether to use symlinks when downloading the model.             **kwargs: Additional keyword arguments to pass to the Llama constructor.         Returns:             A Llama model."""         try:             from huggingface_hub import hf_hub_download, HfFileSystem             from huggingface_hub.utils import validate_repo_id         except ImportError:             raise ImportError(                 "Llama.from_pretrained requires the huggingface-hub package. "                 "You can install it with `pip install huggingface-hub`."             )         validate_repo_id(repo_id)         hffs = HfFileSystem()         files = [             file["name"] if isinstance(file, dict) else file             for file in hffs.ls(repo_id, recursive=True)         ]         # split each file into repo_id, subfolder, filename         file_list: List[str] = []         for file in files:             rel_path = Path(file).relative_to(repo_id)             file_list.append(str(rel_path))         # find the only/first shard file:         matching_files = [file for file in file_list if fnmatch.fnmatch(file, filename)]  # type: ignore         if len(matching_files) == 0:             raise ValueError(                 f"No file found in {repo_id} that match {filename}\n\n"                 f"Available Files:\n{json.dumps(file_list)}"             )         if len(matching_files) > 1:             raise ValueError(                 f"Multiple files found in {repo_id} matching {filename}\n\n"                 f"Available Files:\n{json.dumps(files)}"             )         (matching_file,) = matching_files         subfolder = str(Path(matching_file).parent)         filename = Path(matching_file).name         # download the file         hf_hub_download(             repo_id=repo_id,             filename=filename,             subfolder=subfolder,             local_dir=local_dir,             local_dir_use_symlinks=local_dir_use_symlinks,             cache_dir=cache_dir,         )         if additional_files:             for additonal_file_name in additional_files:                 # find the additional shard file:                 matching_additional_files = [file for file in file_list if fnmatch.fnmatch(file, additonal_file_name)]                 if len(matching_additional_files) == 0:                     raise ValueError(                         f"No file found in {repo_id} that match {additonal_file_name}\n\n"                         f"Available Files:\n{json.dumps(file_list)}"                     )                 if len(matching_additional_files) > 1:                     raise ValueError(                         f"Multiple files found in {repo_id} matching {additonal_file_name}\n\n"                         f"Available Files:\n{json.dumps(files)}"                     )                 (matching_additional_file,) = matching_additional_files                 # download the additional file                 hf_hub_download(                     repo_id=repo_id,                     filename=matching_additional_file,                     subfolder=subfolder,                     local_dir=local_dir,                     local_dir_use_symlinks=local_dir_use_symlinks,                     cache_dir=cache_dir,                 )         if local_dir is None:             model_path = hf_hub_download(                 repo_id=repo_id,                 filename=filename,                 subfolder=subfolder,                 local_dir=local_dir,                 local_dir_use_symlinks=local_dir_use_symlinks,                 cache_dir=cache_dir,                 local_files_only=True,             )         else:             model_path = os.path.join(local_dir, filename)         # loading the first file of a sharded GGUF loads all remaining shard files in the subfolder         return cls(             model_path=model_path,             **kwargs,         )  ``` |
| --- | --- |

#### `__init__(model_path, *, n_gpu_layers=0, split_mode=llama_cpp.LLAMA_SPLIT_MODE_LAYER, main_gpu=0, tensor_split=None, rpc_servers=None, vocab_only=False, use_mmap=True, use_mlock=False, kv_overrides=None, seed=llama_cpp.LLAMA_DEFAULT_SEED, n_ctx=512, n_batch=512, n_ubatch=512, n_threads=None, n_threads_batch=None, rope_scaling_type=llama_cpp.LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED, pooling_type=llama_cpp.LLAMA_POOLING_TYPE_UNSPECIFIED, rope_freq_base=0.0, rope_freq_scale=0.0, yarn_ext_factor=-1.0, yarn_attn_factor=1.0, yarn_beta_fast=32.0, yarn_beta_slow=1.0, yarn_orig_ctx=0, logits_all=False, embedding=False, offload_kqv=True, flash_attn=False, no_perf=False, last_n_tokens_size=64, lora_base=None, lora_scale=1.0, lora_path=None, numa=False, chat_format=None, chat_handler=None, draft_model=None, tokenizer=None, type_k=None, type_v=None, spm_infill=False, verbose=True, **kwargs)`

Load a llama.cpp model from `model_path`.

Examples:

Basic usage

```
>>> import llama_cpp
>>> model = llama_cpp.Llama(
...     model_path="path/to/model",
... )
>>> print(model("The quick brown fox jumps ", stop=["."])["choices"][0]["text"])
the lazy dog

```

Loading a chat model

```
>>> import llama_cpp
>>> model = llama_cpp.Llama(
...     model_path="path/to/model",
...     chat_format="llama-2",
... )
>>> print(model.create_chat_completion(
...     messages=[{
...         "role": "user",
...         "content": "what is the meaning of life?"
...     }]
... ))

```

Parameters:

* **`model_path`**
  (`[str](https://docs.python.org/3/library/stdtypes.html#str)`)
  –

  Path to the model.
* **`n_gpu_layers`**
  (`[int](https://docs.python.org/3/library/functions.html#int)`, default:
  `0`
  )
  –

  Number of layers to offload to GPU (-ngl). If -1, all layers are offloaded.
* **`split_mode`**
  (`[int](https://docs.python.org/3/library/functions.html#int)`, default:
  `[LLAMA_SPLIT_MODE_LAYER](#llama_cpp.llama_cpp.LLAMA_SPLIT_MODE_LAYER "LLAMA_SPLIT_MODE_LAYER = 1

        module-attribute
     (llama_cpp.llama_cpp.LLAMA_SPLIT_MODE_LAYER)")`
  )
  –

  How to split the model across GPUs. See llama\_cpp.LLAMA\_SPLIT\_\* for options.
* **`main_gpu`**
  (`[int](https://docs.python.org/3/library/functions.html#int)`, default:
  `0`
  )
  –

  main\_gpu interpretation depends on split\_mode: LLAMA\_SPLIT\_MODE\_NONE: the GPU that is used for the entire model. LLAMA\_SPLIT\_MODE\_ROW: the GPU that is used for small tensors and intermediate results. LLAMA\_SPLIT\_MODE\_LAYER: ignored
* **`tensor_split`**
  (`[Optional](https://docs.python.org/3/library/typing.html#typing.Optional "llama_cpp.llama_types.Optional")[[List](https://docs.python.org/3/library/typing.html#typing.List "llama_cpp.llama_types.List")[[float](https://docs.python.org/3/library/functions.html#float)]]`, default:
  `None`
  )
  –

  How split tensors should be distributed across GPUs. If None, the model is not split.
* **`rpc_servers`**
  (`[Optional](https://docs.python.org/3/library/typing.html#typing.Optional "llama_cpp.llama_types.Optional")[[str](https://docs.python.org/3/library/stdtypes.html#str)]`, default:
  `None`
  )
  –

  Comma separated list of RPC servers to use for offloading
* **`vocab_only`**
  (`[bool](https://docs.python.org/3/library/functions.html#bool)`, default:
  `False`
  )
  –

  Only load the vocabulary no weights.
* **`use_mmap`**
  (`[bool](https://docs.python.org/3/library/functions.html#bool)`, default:
  `True`
  )
  –

  Use mmap if possible.
* **`use_mlock`**
  (`[bool](https://docs.python.org/3/library/functions.html#bool)`, default:
  `False`
  )
  –

  Force the system to keep the model in RAM.
* **`kv_overrides`**
  (`[Optional](https://docs.python.org/3/library/typing.html#typing.Optional "llama_cpp.llama_types.Optional")[[Dict](https://docs.python.org/3/library/typing.html#typing.Dict "llama_cpp.llama_types.Dict")[[str](https://docs.python.org/3/library/stdtypes.html#str), [Union](https://docs.python.org/3/library/typing.html#typing.Union "llama_cpp.llama_types.Union")[[bool](https://docs.python.org/3/library/functions.html#bool), [int](https://docs.python.org/3/library/functions.html#int), [float](https://docs.python.org/3/library/functions.html#float), [str](https://docs.python.org/3/library/stdtypes.html#str)]]]`, default:
  `None`
  )
  –

  Key-value overrides for the model.
* **`seed`**
  (`[int](https://docs.python.org/3/library/functions.html#int)`, default:
  `[LLAMA_DEFAULT_SEED](#llama_cpp.llama_cpp.LLAMA_DEFAULT_SEED "LLAMA_DEFAULT_SEED = 4294967295

        module-attribute
     (llama_cpp.llama_cpp.LLAMA_DEFAULT_SEED)")`
  )
  –

  RNG seed, -1 for random
* **`n_ctx`**
  (`[int](https://docs.python.org/3/library/functions.html#int)`, default:
  `512`
  )
  –

  Text context, 0 = from model
* **`n_batch`**
  (`[int](https://docs.python.org/3/library/functions.html#int)`, default:
  `512`
  )
  –

  Prompt processing maximum batch size
* **`n_ubatch`**
  (`[int](https://docs.python.org/3/library/functions.html#int)`, default:
  `512`
  )
  –

  Physical batch size
* **`n_threads`**
  (`[Optional](https://docs.python.org/3/library/typing.html#typing.Optional "llama_cpp.llama_types.Optional")[[int](https://docs.python.org/3/library/functions.html#int)]`, default:
  `None`
  )
  –

  Number of threads to use for generation
* **`n_threads_batch`**
  (`[Optional](https://docs.python.org/3/library/typing.html#typing.Optional "llama_cpp.llama_types.Optional")[[int](https://docs.python.org/3/library/functions.html#int)]`, default:
  `None`
  )
  –

  Number of threads to use for batch processing
* **`rope_scaling_type`**
  (`[Optional](https://docs.python.org/3/library/typing.html#typing.Optional "llama_cpp.llama_types.Optional")[[int](https://docs.python.org/3/library/functions.html#int)]`, default:
  `[LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED](#llama_cpp.llama_cpp.LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED "LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED = -1

        module-attribute
     (llama_cpp.llama_cpp.LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED)")`
  )
  –

  RoPE scaling type, from `enum llama_rope_scaling_type`. ref: <https://github.com/ggerganov/llama.cpp/pull/2054>
* **`pooling_type`**
  (`[int](https://docs.python.org/3/library/functions.html#int)`, default:
  `[LLAMA_POOLING_TYPE_UNSPECIFIED](#llama_cpp.llama_cpp.LLAMA_POOLING_TYPE_UNSPECIFIED "LLAMA_POOLING_TYPE_UNSPECIFIED = -1

        module-attribute
     (llama_cpp.llama_cpp.LLAMA_POOLING_TYPE_UNSPECIFIED)")`
  )
  –

  Pooling type, from `enum llama_pooling_type`.
* **`rope_freq_base`**
  (`[float](https://docs.python.org/3/library/functions.html#float)`, default:
  `0.0`
  )
  –

  RoPE base frequency, 0 = from model
* **`rope_freq_scale`**
  (`[float](https://docs.python.org/3/library/functions.html#float)`, default:
  `0.0`
  )
  –

  RoPE frequency scaling factor, 0 = from model
* **`yarn_ext_factor`**
  (`[float](https://docs.python.org/3/library/functions.html#float)`, default:
  `-1.0`
  )
  –

  YaRN extrapolation mix factor, negative = from model
* **`yarn_attn_factor`**
  (`[float](https://docs.python.org/3/library/functions.html#float)`, default:
  `1.0`
  )
  –

  YaRN magnitude scaling factor
* **`yarn_beta_fast`**
  (`[float](https://docs.python.org/3/library/functions.html#float)`, default:
  `32.0`
  )
  –

  YaRN low correction dim
* **`yarn_beta_slow`**
  (`[float](https://docs.python.org/3/library/functions.html#float)`, default:
  `1.0`
  )
  –

  YaRN high correction dim
* **`yarn_orig_ctx`**
  (`[int](https://docs.python.org/3/library/functions.html#int)`, default:
  `0`
  )
  –

  YaRN original context size
* **`logits_all`**
  (`[bool](https://docs.python.org/3/library/functions.html#bool)`, default:
  `False`
  )
  –

  Return logits for all tokens, not just the last token. Must be True for completion to return logprobs.
* **`embedding`**
  (`[bool](https://docs.python.org/3/library/functions.html#bool)`, default:
  `False`
  )
  –

  Embedding mode only.
* **`offload_kqv`**
  (`[bool](https://docs.python.org/3/library/functions.html#bool)`, default:
  `True`
  )
  –

  Offload K, Q, V to GPU.
* **`flash_attn`**
  (`[bool](https://docs.python.org/3/library/functions.html#bool)`, default:
  `False`
  )
  –

  Use flash attention.
* **`no_perf`**
  (`[bool](https://docs.python.org/3/library/functions.html#bool)`, default:
  `False`
  )
  –

  Measure performance timings.
* **`last_n_tokens_size`**
  (`[int](https://docs.python.org/3/library/functions.html#int)`, default:
  `64`
  )
  –

  Maximum number of tokens to keep in the last\_n\_tokens deque.
* **`lora_base`**
  (`[Optional](https://docs.python.org/3/library/typing.html#typing.Optional "llama_cpp.llama_types.Optional")[[str](https://docs.python.org/3/library/stdtypes.html#str)]`, default:
  `None`
  )
  –

  Optional path to base model, useful if using a quantized base model and you want to apply LoRA to an f16 model.
* **`lora_path`**
  (`[Optional](https://docs.python.org/3/library/typing.html#typing.Optional "llama_cpp.llama_types.Optional")[[str](https://docs.python.org/3/library/stdtypes.html#str)]`, default:
  `None`
  )
  –

  Path to a LoRA file to apply to the model.
* **`numa`**
  (`[Union](https://docs.python.org/3/library/typing.html#typing.Union "llama_cpp.llama_types.Union")[[bool](https://docs.python.org/3/library/functions.html#bool), [int](https://docs.python.org/3/library/functions.html#int)]`, default:
  `False`
  )
  –

  numa policy
* **`chat_format`**
  (`[Optional](https://docs.python.org/3/library/typing.html#typing.Optional "llama_cpp.llama_types.Optional")[[str](https://docs.python.org/3/library/stdtypes.html#str)]`, default:
  `None`
  )
  –

  String specifying the chat format to use when calling create\_chat\_completion.
* **`chat_handler`**
  (`[Optional](https://docs.python.org/3/library/typing.html#typing.Optional "llama_cpp.llama_types.Optional")[LlamaChatCompletionHandler]`, default:
  `None`
  )
  –

  Optional chat handler to use when calling create\_chat\_completion.
* **`draft_model`**
  (`[Optional](https://docs.python.org/3/library/typing.html#typing.Optional "llama_cpp.llama_types.Optional")[LlamaDraftModel]`, default:
  `None`
  )
  –

  Optional draft model to use for speculative decoding.
* **`tokenizer`**
  (`[Optional](https://docs.python.org/3/library/typing.html#typing.Optional "llama_cpp.llama_types.Optional")[BaseLlamaTokenizer]`, default:
  `None`
  )
  –

  Optional tokenizer to override the default tokenizer from llama.cpp.
* **`verbose`**
  (`[bool](https://docs.python.org/3/library/functions.html#bool)`, default:
  `True`
  )
  –

  Print verbose output to stderr.
* **`type_k`**
  (`[Optional](https://docs.python.org/3/library/typing.html#typing.Optional "llama_cpp.llama_types.Optional")[[int](https://docs.python.org/3/library/functions.html#int)]`, default:
  `None`
  )
  –

  KV cache data type for K (default: f16)
* **`type_v`**
  (`[Optional](https://docs.python.org/3/library/typing.html#typing.Optional "llama_cpp.llama_types.Optional")[[int](https://docs.python.org/3/library/functions.html#int)]`, default:
  `None`
  )
  –

  KV cache data type for V (default: f16)
* **`spm_infill`**
  (`[bool](https://docs.python.org/3/library/functions.html#bool)`, default:
  `False`
  )
  –

  Use Suffix/Prefix/Middle pattern for infill (instead of Prefix/Suffix/Middle) as some models prefer this.

Raises:

* `[ValueError](https://docs.python.org/3/library/exceptions.html#ValueError)`
  –

  If the model path does not exist.

Returns:

* –

  A Llama instance.| ```                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               ``` | ``` def __init__(     self,     model_path: str,     *,     # Model Params     n_gpu_layers: int = 0,     split_mode: int = llama_cpp.LLAMA_SPLIT_MODE_LAYER,     main_gpu: int = 0,     tensor_split: Optional[List[float]] = None,     rpc_servers: Optional[str] = None,     vocab_only: bool = False,     use_mmap: bool = True,     use_mlock: bool = False,     kv_overrides: Optional[Dict[str, Union[bool, int, float, str]]] = None,     # Context Params     seed: int = llama_cpp.LLAMA_DEFAULT_SEED,     n_ctx: int = 512,     n_batch: int = 512,     n_ubatch: int = 512,     n_threads: Optional[int] = None,     n_threads_batch: Optional[int] = None,     rope_scaling_type: Optional[         int     ] = llama_cpp.LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED,     pooling_type: int = llama_cpp.LLAMA_POOLING_TYPE_UNSPECIFIED,     rope_freq_base: float = 0.0,     rope_freq_scale: float = 0.0,     yarn_ext_factor: float = -1.0,     yarn_attn_factor: float = 1.0,     yarn_beta_fast: float = 32.0,     yarn_beta_slow: float = 1.0,     yarn_orig_ctx: int = 0,     logits_all: bool = False,     embedding: bool = False,     offload_kqv: bool = True,     flash_attn: bool = False,     # Sampling Params     no_perf: bool = False,     last_n_tokens_size: int = 64,     # LoRA Params     lora_base: Optional[str] = None,     lora_scale: float = 1.0,     lora_path: Optional[str] = None,     # Backend Params     numa: Union[bool, int] = False,     # Chat Format Params     chat_format: Optional[str] = None,     chat_handler: Optional[llama_chat_format.LlamaChatCompletionHandler] = None,     # Speculative Decoding     draft_model: Optional[LlamaDraftModel] = None,     # Tokenizer Override     tokenizer: Optional[BaseLlamaTokenizer] = None,     # KV cache quantization     type_k: Optional[int] = None,     type_v: Optional[int] = None,     # Misc     spm_infill: bool = False,     verbose: bool = True,     # Extra Params     **kwargs,  # type: ignore ):     """Load a llama.cpp model from `model_path`.     Examples:         Basic usage         >>> import llama_cpp         >>> model = llama_cpp.Llama(         ...     model_path="path/to/model",         ... )         >>> print(model("The quick brown fox jumps ", stop=["."])["choices"][0]["text"])         the lazy dog         Loading a chat model         >>> import llama_cpp         >>> model = llama_cpp.Llama(         ...     model_path="path/to/model",         ...     chat_format="llama-2",         ... )         >>> print(model.create_chat_completion(         ...     messages=[{         ...         "role": "user",         ...         "content": "what is the meaning of life?"         ...     }]         ... ))     Args:         model_path: Path to the model.         n_gpu_layers: Number of layers to offload to GPU (-ngl). If -1, all layers are offloaded.         split_mode: How to split the model across GPUs. See llama_cpp.LLAMA_SPLIT_* for options.         main_gpu: main_gpu interpretation depends on split_mode: LLAMA_SPLIT_MODE_NONE: the GPU that is used for the entire model. LLAMA_SPLIT_MODE_ROW: the GPU that is used for small tensors and intermediate results. LLAMA_SPLIT_MODE_LAYER: ignored         tensor_split: How split tensors should be distributed across GPUs. If None, the model is not split.         rpc_servers: Comma separated list of RPC servers to use for offloading         vocab_only: Only load the vocabulary no weights.         use_mmap: Use mmap if possible.         use_mlock: Force the system to keep the model in RAM.         kv_overrides: Key-value overrides for the model.         seed: RNG seed, -1 for random         n_ctx: Text context, 0 = from model         n_batch: Prompt processing maximum batch size         n_ubatch: Physical batch size         n_threads: Number of threads to use for generation         n_threads_batch: Number of threads to use for batch processing         rope_scaling_type: RoPE scaling type, from `enum llama_rope_scaling_type`. ref: https://github.com/ggerganov/llama.cpp/pull/2054         pooling_type: Pooling type, from `enum llama_pooling_type`.         rope_freq_base: RoPE base frequency, 0 = from model         rope_freq_scale: RoPE frequency scaling factor, 0 = from model         yarn_ext_factor: YaRN extrapolation mix factor, negative = from model         yarn_attn_factor: YaRN magnitude scaling factor         yarn_beta_fast: YaRN low correction dim         yarn_beta_slow: YaRN high correction dim         yarn_orig_ctx: YaRN original context size         logits_all: Return logits for all tokens, not just the last token. Must be True for completion to return logprobs.         embedding: Embedding mode only.         offload_kqv: Offload K, Q, V to GPU.         flash_attn: Use flash attention.         no_perf: Measure performance timings.         last_n_tokens_size: Maximum number of tokens to keep in the last_n_tokens deque.         lora_base: Optional path to base model, useful if using a quantized base model and you want to apply LoRA to an f16 model.         lora_path: Path to a LoRA file to apply to the model.         numa: numa policy         chat_format: String specifying the chat format to use when calling create_chat_completion.         chat_handler: Optional chat handler to use when calling create_chat_completion.         draft_model: Optional draft model to use for speculative decoding.         tokenizer: Optional tokenizer to override the default tokenizer from llama.cpp.         verbose: Print verbose output to stderr.         type_k: KV cache data type for K (default: f16)         type_v: KV cache data type for V (default: f16)         spm_infill: Use Suffix/Prefix/Middle pattern for infill (instead of Prefix/Suffix/Middle) as some models prefer this.     Raises:         ValueError: If the model path does not exist.     Returns:         A Llama instance.     """     self.verbose = verbose     self._stack = contextlib.ExitStack()     set_verbose(verbose)     if not Llama.__backend_initialized:         with suppress_stdout_stderr(disable=verbose):             llama_cpp.llama_backend_init()         Llama.__backend_initialized = True     if isinstance(numa, bool):         self.numa = (             llama_cpp.GGML_NUMA_STRATEGY_DISTRIBUTE             if numa             else llama_cpp.GGML_NUMA_STRATEGY_DISABLED         )     else:         self.numa = numa     if self.numa != llama_cpp.GGML_NUMA_STRATEGY_DISABLED:         with suppress_stdout_stderr(disable=verbose):             llama_cpp.llama_numa_init(self.numa)     self.model_path = model_path     # Model Params     self.model_params = llama_cpp.llama_model_default_params()     self.model_params.n_gpu_layers = (         0x7FFFFFFF if n_gpu_layers == -1 else n_gpu_layers     )  # 0x7FFFFFFF is INT32 max, will be auto set to all layers     self.model_params.split_mode = split_mode     self.model_params.main_gpu = main_gpu     if rpc_servers is not None:         self.model_params.rpc_servers = rpc_servers.encode("utf-8")         self._rpc_servers = rpc_servers     else:         self._rpc_servers = None     self.tensor_split = tensor_split     self._c_tensor_split = None     if self.tensor_split is not None:         if len(self.tensor_split) > llama_cpp.LLAMA_MAX_DEVICES:             raise ValueError(                 f"Attempt to split tensors that exceed maximum supported devices. Current LLAMA_MAX_DEVICES={llama_cpp.LLAMA_MAX_DEVICES}"             )         # Type conversion and expand the list to the length of LLAMA_MAX_DEVICES         FloatArray = ctypes.c_float * llama_cpp.LLAMA_MAX_DEVICES         self._c_tensor_split = FloatArray(             *tensor_split  # type: ignore         )  # keep a reference to the array so it is not gc'd         self.model_params.tensor_split = self._c_tensor_split     self.model_params.vocab_only = vocab_only     self.model_params.use_mmap = use_mmap if lora_path is None else False     self.model_params.use_mlock = use_mlock     # kv_overrides is the original python dict     self.kv_overrides = kv_overrides     if kv_overrides is not None:         # _kv_overrides_array is a ctypes.Array of llama_model_kv_override Structs         kvo_array_len = len(kv_overrides) + 1  # for sentinel element         self._kv_overrides_array = (             llama_cpp.llama_model_kv_override * kvo_array_len         )()         for i, (k, v) in enumerate(kv_overrides.items()):             self._kv_overrides_array[i].key = k.encode("utf-8")             if isinstance(v, bool):                 self._kv_overrides_array[                     i                 ].tag = llama_cpp.LLAMA_KV_OVERRIDE_TYPE_BOOL                 self._kv_overrides_array[i].value.val_bool = v             elif isinstance(v, int):                 self._kv_overrides_array[                     i                 ].tag = llama_cpp.LLAMA_KV_OVERRIDE_TYPE_INT                 self._kv_overrides_array[i].value.val_i64 = v             elif isinstance(v, float):                 self._kv_overrides_array[                     i                 ].tag = llama_cpp.LLAMA_KV_OVERRIDE_TYPE_FLOAT                 self._kv_overrides_array[i].value.val_f64 = v             elif isinstance(v, str):  # type: ignore                 v_bytes = v.encode("utf-8")                 if len(v_bytes) > 128:  # TODO: Make this a constant                     raise ValueError(f"Value for {k} is too long: {v}")                 v_bytes = v_bytes.ljust(128, b"\0")                 self._kv_overrides_array[                     i                 ].tag = llama_cpp.LLAMA_KV_OVERRIDE_TYPE_STR                 # copy min(v_bytes, 128) to str_value                 address = typing.cast(                     int,                     ctypes.addressof(self._kv_overrides_array[i].value)                     + llama_cpp.llama_model_kv_override_value.val_str.offset,                 )                 buffer_start = ctypes.cast(address, ctypes.POINTER(ctypes.c_char))                 ctypes.memmove(                     buffer_start,                     v_bytes,                     128,                 )             else:                 raise ValueError(f"Unknown value type for {k}: {v}")         self._kv_overrides_array[             -1         ].key = b"\0"  # ensure sentinel element is zeroed         self.model_params.kv_overrides = self._kv_overrides_array     self.n_batch = min(n_ctx, n_batch)  # ???     self.n_threads = n_threads or max(multiprocessing.cpu_count() // 2, 1)     self.n_threads_batch = n_threads_batch or multiprocessing.cpu_count()     # Used by the sampler     self._seed = seed or llama_cpp.LLAMA_DEFAULT_SEED     # Context Params     self.context_params = llama_cpp.llama_context_default_params()     self.context_params.n_ctx = n_ctx     self.context_params.n_batch = self.n_batch     self.context_params.n_ubatch = min(self.n_batch, n_ubatch)     self.context_params.n_threads = self.n_threads     self.context_params.n_threads_batch = self.n_threads_batch     self.context_params.rope_scaling_type = (         rope_scaling_type         if rope_scaling_type is not None         else llama_cpp.LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED     )     self.context_params.pooling_type = pooling_type     self.context_params.rope_freq_base = (         rope_freq_base if rope_freq_base != 0.0 else 0     )     self.context_params.rope_freq_scale = (         rope_freq_scale if rope_freq_scale != 0.0 else 0     )     self.context_params.yarn_ext_factor = (         yarn_ext_factor if yarn_ext_factor != 0.0 else 0     )     self.context_params.yarn_attn_factor = (         yarn_attn_factor if yarn_attn_factor != 0.0 else 0     )     self.context_params.yarn_beta_fast = (         yarn_beta_fast if yarn_beta_fast != 0.0 else 0     )     self.context_params.yarn_beta_slow = (         yarn_beta_slow if yarn_beta_slow != 0.0 else 0     )     self.context_params.yarn_orig_ctx = yarn_orig_ctx if yarn_orig_ctx != 0 else 0     self.context_params.logits_all = (         logits_all if draft_model is None else True     )  # Must be set to True for speculative decoding     self.context_params.embeddings = embedding  # TODO: Rename to embeddings     self.context_params.offload_kqv = offload_kqv     self.context_params.flash_attn = flash_attn     #  KV cache quantization     if type_k is not None:         self.context_params.type_k = type_k     if type_v is not None:         self.context_params.type_v = type_v     # Sampling Params     self.context_params.no_perf = no_perf     self.last_n_tokens_size = last_n_tokens_size     self.cache: Optional[BaseLlamaCache] = None     self.lora_base = lora_base     self.lora_scale = lora_scale     self.lora_path = lora_path     self.spm_infill = spm_infill     if not os.path.exists(model_path):         raise ValueError(f"Model path does not exist: {model_path}")     self._model = self._stack.enter_context(         contextlib.closing(             internals.LlamaModel(                 path_model=self.model_path,                 params=self.model_params,                 verbose=self.verbose,             )         )     )     # Override tokenizer     self.tokenizer_ = tokenizer or LlamaTokenizer(self)     # Set the default value for the context and correct the batch     if n_ctx == 0:         n_ctx = self._model.n_ctx_train()         self.n_batch = min(n_ctx, n_batch)         self.context_params.n_ctx = self._model.n_ctx_train()         self.context_params.n_batch = self.n_batch         self.context_params.n_ubatch = min(self.n_batch, n_ubatch)     self._ctx = self._stack.enter_context(         contextlib.closing(             internals.LlamaContext(                 model=self._model,                 params=self.context_params,                 verbose=self.verbose,             )         )     )     self._batch = self._stack.enter_context(         contextlib.closing(             internals.LlamaBatch(                 n_tokens=self.n_batch,                 embd=0,                 n_seq_max=self.context_params.n_ctx,                 verbose=self.verbose,             )         )     )     self._lora_adapter: Optional[llama_cpp.llama_adapter_lora_p] = None     if self.lora_path:         self._lora_adapter = llama_cpp.llama_adapter_lora_init(             self._model.model,             self.lora_path.encode("utf-8"),         )         if self._lora_adapter is None:             raise RuntimeError(                 f"Failed to initialize LoRA adapter from lora path: {self.lora_path}"             )         def free_lora_adapter():             if self._lora_adapter is None:                 return             llama_cpp.llama_adapter_lora_free(self._lora_adapter)             self._lora_adapter = None         self._stack.callback(free_lora_adapter)         if llama_cpp.llama_set_adapter_lora(             self._ctx.ctx, self._lora_adapter, self.lora_scale         ):             raise RuntimeError(                 f"Failed to set LoRA adapter from lora path: {self.lora_path}"             )     if self.verbose:         print(llama_cpp.llama_print_system_info().decode("utf-8"), file=sys.stderr)     self.chat_format = chat_format     self.chat_handler = chat_handler     self._chat_handlers: Dict[         str, llama_chat_format.LlamaChatCompletionHandler     ] = {}     self.draft_model = draft_model     self._n_vocab = self.n_vocab()     self._n_ctx = self.n_ctx()     self._token_nl = self.token_nl()     self._token_eos = self.token_eos()     self._candidates = internals.LlamaTokenDataArray(n_vocab=self._n_vocab)     self.n_tokens = 0     self.input_ids: npt.NDArray[np.intc] = np.ndarray((n_ctx,), dtype=np.intc)     self.scores: npt.NDArray[np.single] = np.ndarray(         (n_ctx if logits_all == True else n_batch, self._n_vocab), dtype=np.single     )     self._mirostat_mu = ctypes.c_float(         2.0 * 5.0     )  # TODO: Move this to sampling context     try:         self.metadata = self._model.metadata()     except Exception as e:         self.metadata = {}         if self.verbose:             print(f"Failed to load metadata: {e}", file=sys.stderr)     if self.verbose:         print(f"Model metadata: {self.metadata}", file=sys.stderr)     eos_token_id = self.token_eos()     bos_token_id = self.token_bos()     eos_token = (         self._model.token_get_text(eos_token_id) if eos_token_id != -1 else ""     )     bos_token = (         self._model.token_get_text(bos_token_id) if bos_token_id != -1 else ""     )     # Unfortunately the llama.cpp API does not return metadata arrays, so we can't get template names from tokenizer.chat_templates     template_choices = dict(         (name[10:], template)         for name, template in self.metadata.items()         if name.startswith("tokenizer.chat_template.")     )     if "tokenizer.chat_template" in self.metadata:         template_choices["chat_template.default"] = self.metadata[             "tokenizer.chat_template"         ]     if self.verbose and template_choices:         print(             f"Available chat formats from metadata: {', '.join(template_choices.keys())}",             file=sys.stderr,         )     for name, template in template_choices.items():         self._chat_handlers[name] = llama_chat_format.Jinja2ChatFormatter(             template=template,             eos_token=eos_token,             bos_token=bos_token,             stop_token_ids=[eos_token_id],         ).to_chat_handler()     if (         self.chat_format is None         and self.chat_handler is None         and "chat_template.default" in template_choices     ):         chat_format = llama_chat_format.guess_chat_format_from_gguf_metadata(             self.metadata         )         if chat_format is not None:             self.chat_format = chat_format             if self.verbose:                 print(f"Guessed chat format: {chat_format}", file=sys.stderr)         else:             if self.verbose:                 print(                     f"Using gguf chat template: {template_choices['chat_template.default']}",                     file=sys.stderr,                 )                 print(f"Using chat eos_token: {eos_token}", file=sys.stderr)                 print(f"Using chat bos_token: {bos_token}", file=sys.stderr)             self.chat_format = "chat_template.default"     if self.chat_format is None and self.chat_handler is None:         self.chat_format = "llama-2"         if self.verbose:             print(                 f"Using fallback chat format: {self.chat_format}", file=sys.stderr             )     self._sampler = None  ``` |
| --- | --- |

#### `tokenize(text, add_bos=True, special=False)`

Tokenize a string.

Parameters:

* **`text`**
  (`[bytes](https://docs.python.org/3/library/stdtypes.html#bytes)`)
  –

  The utf-8 encoded string to tokenize.
* **`add_bos`**
  (`[bool](https://docs.python.org/3/library/functions.html#bool)`, default:
  `True`
  )
  –

  Whether to add a beginning of sequence token.
* **`special`**
  (`[bool](https://docs.python.org/3/library/functions.html#bool)`, default:
  `False`
  )
  –

  Whether to tokenize special tokens.

Raises:

* `[RuntimeError](https://docs.python.org/3/library/exceptions.html#RuntimeError)`
  –

  If the tokenization failed.

Returns:

* `[List](https://docs.python.org/3/library/typing.html#typing.List "llama_cpp.llama_types.List")[[int](https://docs.python.org/3/library/functions.html#int)]`
  –

  A list of tokens.

#### `detokenize(tokens, prev_tokens=None, special=False)`

Detokenize a list of tokens.

Parameters:

* **`tokens`**
  (`[List](https://docs.python.org/3/library/typing.html#typing.List "llama_cpp.llama_types.List")[[int](https://docs.python.org/3/library/functions.html#int)]`)
  –

  The list of tokens to detokenize.
* **`prev_tokens`**
  (`[Optional](https://docs.python.org/3/library/typing.html#typing.Optional "llama_cpp.llama_types.Optional")[[List](https://docs.python.org/3/library/typing.html#typing.List "llama_cpp.llama_types.List")[[int](https://docs.python.org/3/library/functions.html#int)]]`, default:
  `None`
  )
  –

  The list of previous tokens. Offset mapping will be performed if provided.
* **`special`**
  (`[bool](https://docs.python.org/3/library/functions.html#bool)`, default:
  `False`
  )
  –

  Whether to detokenize special tokens.

Returns:

* `[bytes](https://docs.python.org/3/library/stdtypes.html#bytes)`
  –

  The detokenized string.

#### `reset()`

Reset the model state.

#### `eval(tokens)`

Evaluate a list of tokens.

Parameters:

* **`tokens`**
  (`[Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence "typing.Sequence")[[int](https://docs.python.org/3/library/functions.html#int)]`)
  –

  The list of tokens to evaluate.

#### `sample(top_k=40, top_p=0.95, min_p=0.05, typical_p=1.0, temp=0.8, repeat_penalty=1.0, frequency_penalty=0.0, presence_penalty=0.0, tfs_z=1.0, mirostat_mode=0, mirostat_eta=0.1, mirostat_tau=5.0, penalize_nl=True, logits_processor=None, grammar=None, idx=None)`

Sample a token from the model.

Parameters:

* **`top_k`**
  (`[int](https://docs.python.org/3/library/functions.html#int)`, default:
  `40`
  )
  –

  The top-k sampling parameter.
* **`top_p`**
  (`[float](https://docs.python.org/3/library/functions.html#float)`, default:
  `0.95`
  )
  –

  The top-p sampling parameter.
* **`temp`**
  (`[float](https://docs.python.org/3/library/functions.html#float)`, default:
  `0.8`
  )
  –

  The temperature parameter.
* **`repeat_penalty`**
  (`[float](https://docs.python.org/3/library/functions.html#float)`, default:
  `1.0`
  )
  –

  The repeat penalty parameter.

Returns:

* –

  The sampled token.

#### `generate(tokens, top_k=40, top_p=0.95, min_p=0.05, typical_p=1.0, temp=0.8, repeat_penalty=1.0, reset=True, frequency_penalty=0.0, presence_penalty=0.0, tfs_z=1.0, mirostat_mode=0, mirostat_tau=5.0, mirostat_eta=0.1, penalize_nl=True, logits_processor=None, stopping_criteria=None, grammar=None)`

Create a generator of tokens from a prompt.

Examples:

```
>>> llama = Llama("models/ggml-7b.bin")
>>> tokens = llama.tokenize(b"Hello, world!")
>>> for token in llama.generate(tokens, top_k=40, top_p=0.95, temp=1.0, repeat_penalty=1.0):
...     print(llama.detokenize([token]))

```

Parameters:

* **`tokens`**
  (`[Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence "typing.Sequence")[[int](https://docs.python.org/3/library/functions.html#int)]`)
  –

  The prompt tokens.
* **`top_k`**
  (`[int](https://docs.python.org/3/library/functions.html#int)`, default:
  `40`
  )
  –

  The top-k sampling parameter.
* **`top_p`**
  (`[float](https://docs.python.org/3/library/functions.html#float)`, default:
  `0.95`
  )
  –

  The top-p sampling parameter.
* **`temp`**
  (`[float](https://docs.python.org/3/library/functions.html#float)`, default:
  `0.8`
  )
  –

  The temperature parameter.
* **`repeat_penalty`**
  (`[float](https://docs.python.org/3/library/functions.html#float)`, default:
  `1.0`
  )
  –

  The repeat penalty parameter.
* **`reset`**
  (`[bool](https://docs.python.org/3/library/functions.html#bool)`, default:
  `True`
  )
  –

  Whether to reset the model state.

Yields:

* `[int](https://docs.python.org/3/library/functions.html#int)`
  –

  The generated tokens.

#### `create_embedding(input, model=None)`

Embed a string.

Parameters:

* **`input`**
  (`[Union](https://docs.python.org/3/library/typing.html#typing.Union "llama_cpp.llama_types.Union")[[str](https://docs.python.org/3/library/stdtypes.html#str), [List](https://docs.python.org/3/library/typing.html#typing.List "llama_cpp.llama_types.List")[[str](https://docs.python.org/3/library/stdtypes.html#str)]]`)
  –

  The utf-8 encoded string to embed.

Returns:

* `[CreateEmbeddingResponse](#llama_cpp.llama_types.CreateEmbeddingResponse "CreateEmbeddingResponse (llama_cpp.llama_types.CreateEmbeddingResponse)")`
  –

  An embedding object.

#### `embed(input, normalize=False, truncate=True, return_count=False)`

Embed a string.

Parameters:

* **`input`**
  (`[Union](https://docs.python.org/3/library/typing.html#typing.Union "llama_cpp.llama_types.Union")[[str](https://docs.python.org/3/library/stdtypes.html#str), [List](https://docs.python.org/3/library/typing.html#typing.List "llama_cpp.llama_types.List")[[str](https://docs.python.org/3/library/stdtypes.html#str)]]`)
  –

  The utf-8 encoded string to embed.

Returns:

* –

  A list of embeddings

#### `create_completion(prompt, suffix=None, max_tokens=16, temperature=0.8, top_p=0.95, min_p=0.05, typical_p=1.0, logprobs=None, echo=False, stop=[], frequency_penalty=0.0, presence_penalty=0.0, repeat_penalty=1.0, top_k=40, stream=False, seed=None, tfs_z=1.0, mirostat_mode=0, mirostat_tau=5.0, mirostat_eta=0.1, model=None, stopping_criteria=None, logits_processor=None, grammar=None, logit_bias=None)`

Generate text from a prompt.

Parameters:

* **`prompt`**
  (`[Union](https://docs.python.org/3/library/typing.html#typing.Union "llama_cpp.llama_types.Union")[[str](https://docs.python.org/3/library/stdtypes.html#str), [List](https://docs.python.org/3/library/typing.html#typing.List "llama_cpp.llama_types.List")[[int](https://docs.python.org/3/library/functions.html#int)]]`)
  –

  The prompt to generate text from.
* **`suffix`**
  (`[Optional](https://docs.python.org/3/library/typing.html#typing.Optional "llama_cpp.llama_types.Optional")[[str](https://docs.python.org/3/library/stdtypes.html#str)]`, default:
  `None`
  )
  –

  A suffix to append to the generated text. If None, no suffix is appended.
* **`max_tokens`**
  (`[Optional](https://docs.python.org/3/library/typing.html#typing.Optional "llama_cpp.llama_types.Optional")[[int](https://docs.python.org/3/library/functions.html#int)]`, default:
  `16`
  )
  –

  The maximum number of tokens to generate. If max\_tokens <= 0 or None, the maximum number of tokens to generate is unlimited and depends on n\_ctx.
* **`temperature`**
  (`[float](https://docs.python.org/3/library/functions.html#float)`, default:
  `0.8`
  )
  –

  The temperature to use for sampling.
* **`top_p`**
  (`[float](https://docs.python.org/3/library/functions.html#float)`, default:
  `0.95`
  )
  –

  The top-p value to use for nucleus sampling. Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" <https://arxiv.org/abs/1904.09751>
* **`min_p`**
  (`[float](https://docs.python.org/3/library/functions.html#float)`, default:
  `0.05`
  )
  –

  The min-p value to use for minimum p sampling. Minimum P sampling as described in <https://github.com/ggerganov/llama.cpp/pull/3841>
* **`typical_p`**
  (`[float](https://docs.python.org/3/library/functions.html#float)`, default:
  `1.0`
  )
  –

  The typical-p value to use for sampling. Locally Typical Sampling implementation described in the paper <https://arxiv.org/abs/2202.00666>.
* **`logprobs`**
  (`[Optional](https://docs.python.org/3/library/typing.html#typing.Optional "llama_cpp.llama_types.Optional")[[int](https://docs.python.org/3/library/functions.html#int)]`, default:
  `None`
  )
  –

  The number of logprobs to return. If None, no logprobs are returned.
* **`echo`**
  (`[bool](https://docs.python.org/3/library/functions.html#bool)`, default:
  `False`
  )
  –

  Whether to echo the prompt.
* **`stop`**
  (`[Optional](https://docs.python.org/3/library/typing.html#typing.Optional "llama_cpp.llama_types.Optional")[[Union](https://docs.python.org/3/library/typing.html#typing.Union "llama_cpp.llama_types.Union")[[str](https://docs.python.org/3/library/stdtypes.html#str), [List](https://docs.python.org/3/library/typing.html#typing.List "llama_cpp.llama_types.List")[[str](https://docs.python.org/3/library/stdtypes.html#str)]]]`, default:
  `[]`
  )
  –

  A list of strings to stop generation when encountered.
* **`frequency_penalty`**
  (`[float](https://docs.python.org/3/library/functions.html#float)`, default:
  `0.0`
  )
  –

  The penalty to apply to tokens based on their frequency in the prompt.
* **`presence_penalty`**
  (`[float](https://docs.python.org/3/library/functions.html#float)`, default:
  `0.0`
  )
  –

  The penalty to apply to tokens based on their presence in the prompt.
* **`repeat_penalty`**
  (`[float](https://docs.python.org/3/library/functions.html#float)`, default:
  `1.0`
  )
  –

  The penalty to apply to repeated tokens.
* **`top_k`**
  (`[int](https://docs.python.org/3/library/functions.html#int)`, default:
  `40`
  )
  –

  The top-k value to use for sampling. Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" <https://arxiv.org/abs/1904.09751>
* **`stream`**
  (`[bool](https://docs.python.org/3/library/functions.html#bool)`, default:
  `False`
  )
  –

  Whether to stream the results.
* **`seed`**
  (`[Optional](https://docs.python.org/3/library/typing.html#typing.Optional "llama_cpp.llama_types.Optional")[[int](https://docs.python.org/3/library/functions.html#int)]`, default:
  `None`
  )
  –

  The seed to use for sampling.
* **`tfs_z`**
  (`[float](https://docs.python.org/3/library/functions.html#float)`, default:
  `1.0`
  )
  –

  The tail-free sampling parameter. Tail Free Sampling described in <https://www.trentonbricken.com/Tail-Free-Sampling/>.
* **`mirostat_mode`**
  (`[int](https://docs.python.org/3/library/functions.html#int)`, default:
  `0`
  )
  –

  The mirostat sampling mode.
* **`mirostat_tau`**
  (`[float](https://docs.python.org/3/library/functions.html#float)`, default:
  `5.0`
  )
  –

  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
* **`mirostat_eta`**
  (`[float](https://docs.python.org/3/library/functions.html#float)`, default:
  `0.1`
  )
  –

  The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
* **`model`**
  (`[Optional](https://docs.python.org/3/library/typing.html#typing.Optional "llama_cpp.llama_types.Optional")[[str](https://docs.python.org/3/library/stdtypes.html#str)]`, default:
  `None`
  )
  –

  The name to use for the model in the completion object.
* **`stopping_criteria`**
  (`[Optional](https://docs.python.org/3/library/typing.html#typing.Optional "llama_cpp.llama_types.Optional")[[StoppingCriteriaList](#llama_cpp.StoppingCriteriaList "llama_cpp.StoppingCriteriaList (llama_cpp.llama.StoppingCriteriaList)")]`, default:
  `None`
  )
  –

  A list of stopping criteria to use.
* **`logits_processor`**
  (`[Optional](https://docs.python.org/3/library/typing.html#typing.Optional "llama_cpp.llama_types.Optional")[[LogitsProcessorList](#llama_cpp.LogitsProcessorList "llama_cpp.LogitsProcessorList (llama_cpp.llama.LogitsProcessorList)")]`, default:
  `None`
  )
  –

  A list of logits processors to use.
* **`grammar`**
  (`[Optional](https://docs.python.org/3/library/typing.html#typing.Optional "llama_cpp.llama_types.Optional")[[LlamaGrammar](#llama_cpp.LlamaGrammar "llama_cpp.LlamaGrammar (llama_cpp.llama_grammar.LlamaGrammar)")]`, default:
  `None`
  )
  –

  A grammar to use for constrained sampling.
* **`logit_bias`**
  (`[Optional](https://docs.python.org/3/library/typing.html#typing.Optional "llama_cpp.llama_types.Optional")[[Dict](https://docs.python.org/3/library/typing.html#typing.Dict "llama_cpp.llama_types.Dict")[[int](https://docs.python.org/3/library/functions.html#int), [float](https://docs.python.org/3/library/functions.html#float)]]`, default:
  `None`
  )
  –

  A logit bias to use.

Raises:

* `[ValueError](https://docs.python.org/3/library/exceptions.html#ValueError)`
  –

  If the requested tokens exceed the context window.
* `[RuntimeError](https://docs.python.org/3/library/exceptions.html#RuntimeError)`
  –

  If the prompt fails to tokenize or the model fails to evaluate the prompt.

Returns:

* `[Union](https://docs.python.org/3/library/typing.html#typing.Union "llama_cpp.llama_types.Union")[[CreateCompletionResponse](#llama_cpp.llama_types.CreateCompletionResponse "CreateCompletionResponse (llama_cpp.llama_types.CreateCompletionResponse)"), [Iterator](https://docs.python.org/3/library/typing.html#typing.Iterator "typing.Iterator")[[CreateCompletionStreamResponse](#llama_cpp.llama_types.CreateCompletionStreamResponse "CreateCompletionStreamResponse = CreateCompletionResponse

        module-attribute
     (llama_cpp.llama_types.CreateCompletionStreamResponse)")]]`
  –

  Response object containing the generated text.

#### `__call__(prompt, suffix=None, max_tokens=16, temperature=0.8, top_p=0.95, min_p=0.05, typical_p=1.0, logprobs=None, echo=False, stop=[], frequency_penalty=0.0, presence_penalty=0.0, repeat_penalty=1.0, top_k=40, stream=False, seed=None, tfs_z=1.0, mirostat_mode=0, mirostat_tau=5.0, mirostat_eta=0.1, model=None, stopping_criteria=None, logits_processor=None, grammar=None, logit_bias=None)`

Generate text from a prompt.

Parameters:

* **`prompt`**
  (`[str](https://docs.python.org/3/library/stdtypes.html#str)`)
  –

  The prompt to generate text from.
* **`suffix`**
  (`[Optional](https://docs.python.org/3/library/typing.html#typing.Optional "llama_cpp.llama_types.Optional")[[str](https://docs.python.org/3/library/stdtypes.html#str)]`, default:
  `None`
  )
  –

  A suffix to append to the generated text. If None, no suffix is appended.
* **`max_tokens`**
  (`[Optional](https://docs.python.org/3/library/typing.html#typing.Optional "llama_cpp.llama_types.Optional")[[int](https://docs.python.org/3/library/functions.html#int)]`, default:
  `16`
  )
  –

  The maximum number of tokens to generate. If max\_tokens <= 0 or None, the maximum number of tokens to generate is unlimited and depends on n\_ctx.
* **`temperature`**
  (`[float](https://docs.python.org/3/library/functions.html#float)`, default:
  `0.8`
  )
  –

  The temperature to use for sampling.
* **`top_p`**
  (`[float](https://docs.python.org/3/library/functions.html#float)`, default:
  `0.95`
  )
  –

  The top-p value to use for nucleus sampling. Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" <https://arxiv.org/abs/1904.09751>
* **`min_p`**
  (`[float](https://docs.python.org/3/library/functions.html#float)`, default:
  `0.05`
  )
  –

  The min-p value to use for minimum p sampling. Minimum P sampling as described in <https://github.com/ggerganov/llama.cpp/pull/3841>
* **`typical_p`**
  (`[float](https://docs.python.org/3/library/functions.html#float)`, default:
  `1.0`
  )
  –

  The typical-p value to use for sampling. Locally Typical Sampling implementation described in the paper <https://arxiv.org/abs/2202.00666>.
* **`logprobs`**
  (`[Optional](https://docs.python.org/3/library/typing.html#typing.Optional "llama_cpp.llama_types.Optional")[[int](https://docs.python.org/3/library/functions.html#int)]`, default:
  `None`
  )
  –

  The number of logprobs to return. If None, no logprobs are returned.
* **`echo`**
  (`[bool](https://docs.python.org/3/library/functions.html#bool)`, default:
  `False`
  )
  –

  Whether to echo the prompt.
* **`stop`**
  (`[Optional](https://docs.python.org/3/library/typing.html#typing.Optional "llama_cpp.llama_types.Optional")[[Union](https://docs.python.org/3/library/typing.html#typing.Union "llama_cpp.llama_types.Union")[[str](https://docs.python.org/3/library/stdtypes.html#str), [List](https://docs.python.org/3/library/typing.html#typing.List "llama_cpp.llama_types.List")[[str](https://docs.python.org/3/library/stdtypes.html#str)]]]`, default:
  `[]`
  )
  –

  A list of strings to stop generation when encountered.
* **`frequency_penalty`**
  (`[float](https://docs.python.org/3/library/functions.html#float)`, default:
  `0.0`
  )
  –

  The penalty to apply to tokens based on their frequency in the prompt.
* **`presence_penalty`**
  (`[float](https://docs.python.org/3/library/functions.html#float)`, default:
  `0.0`
  )
  –

  The penalty to apply to tokens based on their presence in the prompt.
* **`repeat_penalty`**
  (`[float](https://docs.python.org/3/library/functions.html#float)`, default:
  `1.0`
  )
  –

  The penalty to apply to repeated tokens.
* **`top_k`**
  (`[int](https://docs.python.org/3/library/functions.html#int)`, default:
  `40`
  )
  –

  The top-k value to use for sampling. Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" <https://arxiv.org/abs/1904.09751>
* **`stream`**
  (`[bool](https://docs.python.org/3/library/functions.html#bool)`, default:
  `False`
  )
  –

  Whether to stream the results.
* **`seed`**
  (`[Optional](https://docs.python.org/3/library/typing.html#typing.Optional "llama_cpp.llama_types.Optional")[[int](https://docs.python.org/3/library/functions.html#int)]`, default:
  `None`
  )
  –

  The seed to use for sampling.
* **`tfs_z`**
  (`[float](https://docs.python.org/3/library/functions.html#float)`, default:
  `1.0`
  )
  –

  The tail-free sampling parameter. Tail Free Sampling described in <https://www.trentonbricken.com/Tail-Free-Sampling/>.
* **`mirostat_mode`**
  (`[int](https://docs.python.org/3/library/functions.html#int)`, default:
  `0`
  )
  –

  The mirostat sampling mode.
* **`mirostat_tau`**
  (`[float](https://docs.python.org/3/library/functions.html#float)`, default:
  `5.0`
  )
  –

  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
* **`mirostat_eta`**
  (`[float](https://docs.python.org/3/library/functions.html#float)`, default:
  `0.1`
  )
  –

  The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
* **`model`**
  (`[Optional](https://docs.python.org/3/library/typing.html#typing.Optional "llama_cpp.llama_types.Optional")[[str](https://docs.python.org/3/library/stdtypes.html#str)]`, default:
  `None`
  )
  –

  The name to use for the model in the completion object.
* **`stopping_criteria`**
  (`[Optional](https://docs.python.org/3/library/typing.html#typing.Optional "llama_cpp.llama_types.Optional")[[StoppingCriteriaList](#llama_cpp.StoppingCriteriaList "llama_cpp.StoppingCriteriaList (llama_cpp.llama.StoppingCriteriaList)")]`, default:
  `None`
  )
  –

  A list of stopping criteria to use.
* **`logits_processor`**
  (`[Optional](https://docs.python.org/3/library/typing.html#typing.Optional "llama_cpp.llama_types.Optional")[[LogitsProcessorList](#llama_cpp.LogitsProcessorList "llama_cpp.LogitsProcessorList (llama_cpp.llama.LogitsProcessorList)")]`, default:
  `None`
  )
  –

  A list of logits processors to use.
* **`grammar`**
  (`[Optional](https://docs.python.org/3/library/typing.html#typing.Optional "llama_cpp.llama_types.Optional")[[LlamaGrammar](#llama_cpp.LlamaGrammar "llama_cpp.LlamaGrammar (llama_cpp.llama_grammar.LlamaGrammar)")]`, default:
  `None`
  )
  –

  A grammar to use for constrained sampling.
* **`logit_bias`**
  (`[Optional](https://docs.python.org/3/library/typing.html#typing.Optional "llama_cpp.llama_types.Optional")[[Dict](https://docs.python.org/3/library/typing.html#typing.Dict "llama_cpp.llama_types.Dict")[[int](https://docs.python.org/3/library/functions.html#int), [float](https://docs.python.org/3/library/functions.html#float)]]`, default:
  `None`
  )
  –

  A logit bias to use.

Raises:

* `[ValueError](https://docs.python.org/3/library/exceptions.html#ValueError)`
  –

  If the requested tokens exceed the context window.
* `[RuntimeError](https://docs.python.org/3/library/exceptions.html#RuntimeError)`
  –

  If the prompt fails to tokenize or the model fails to evaluate the prompt.

Returns:

* `[Union](https://docs.python.org/3/library/typing.html#typing.Union "llama_cpp.llama_types.Union")[[CreateCompletionResponse](#llama_cpp.llama_types.CreateCompletionResponse "CreateCompletionResponse (llama_cpp.llama_types.CreateCompletionResponse)"), [Iterator](https://docs.python.org/3/library/typing.html#typing.Iterator "typing.Iterator")[[CreateCompletionStreamResponse](#llama_cpp.llama_types.CreateCompletionStreamResponse "CreateCompletionStreamResponse = CreateCompletionResponse

        module-attribute
     (llama_cpp.llama_types.CreateCompletionStreamResponse)")]]`
  –

  Response object containing the generated text.

#### `create_chat_completion(messages, functions=None, function_call=None, tools=None, tool_choice=None, temperature=0.2, top_p=0.95, top_k=40, min_p=0.05, typical_p=1.0, stream=False, stop=[], seed=None, response_format=None, max_tokens=None, presence_penalty=0.0, frequency_penalty=0.0, repeat_penalty=1.0, tfs_z=1.0, mirostat_mode=0, mirostat_tau=5.0, mirostat_eta=0.1, model=None, logits_processor=None, grammar=None, logit_bias=None, logprobs=None, top_logprobs=None)`

Generate a chat completion from a list of messages.

Parameters:

* **`messages`**
  (`[List](https://docs.python.org/3/library/typing.html#typing.List "llama_cpp.llama_types.List")[[ChatCompletionRequestMessage](#llama_cpp.llama_types.ChatCompletionRequestMessage "ChatCompletionRequestMessage = Union[ChatCompletionRequestSystemMessage, ChatCompletionRequestUserMessage, ChatCompletionRequestAssistantMessage, ChatCompletionRequestUserMessage, ChatCompletionRequestToolMessage, ChatCompletionRequestFunctionMessage]

        module-attribute
     (llama_cpp.llama_types.ChatCompletionRequestMessage)")]`)
  –

  A list of messages to generate a response for.
* **`functions`**
  (`[Optional](https://docs.python.org/3/library/typing.html#typing.Optional "llama_cpp.llama_types.Optional")[[List](https://docs.python.org/3/library/typing.html#typing.List "llama_cpp.llama_types.List")[[ChatCompletionFunction](#llama_cpp.llama_types.ChatCompletionFunction "ChatCompletionFunction (llama_cpp.llama_types.ChatCompletionFunction)")]]`, default:
  `None`
  )
  –

  A list of functions to use for the chat completion.
* **`function_call`**
  (`[Optional](https://docs.python.org/3/library/typing.html#typing.Optional "llama_cpp.llama_types.Optional")[[ChatCompletionRequestFunctionCall](#llama_cpp.llama_types.ChatCompletionRequestFunctionCall "ChatCompletionRequestFunctionCall = Union[Literal['none', 'auto'], ChatCompletionRequestFunctionCallOption]

        module-attribute
     (llama_cpp.llama_types.ChatCompletionRequestFunctionCall)")]`, default:
  `None`
  )
  –

  A function call to use for the chat completion.
* **`tools`**
  (`[Optional](https://docs.python.org/3/library/typing.html#typing.Optional "llama_cpp.llama_types.Optional")[[List](https://docs.python.org/3/library/typing.html#typing.List "llama_cpp.llama_types.List")[[ChatCompletionTool](#llama_cpp.llama_types.ChatCompletionTool "ChatCompletionTool (llama_cpp.llama_types.ChatCompletionTool)")]]`, default:
  `None`
  )
  –

  A list of tools to use for the chat completion.
* **`tool_choice`**
  (`[Optional](https://docs.python.org/3/library/typing.html#typing.Optional "llama_cpp.llama_types.Optional")[[ChatCompletionToolChoiceOption](#llama_cpp.llama_types.ChatCompletionToolChoiceOption "ChatCompletionToolChoiceOption = Union[Literal['none', 'auto', 'required'], ChatCompletionNamedToolChoice]

        module-attribute
     (llama_cpp.llama_types.ChatCompletionToolChoiceOption)")]`, default:
  `None`
  )
  –

  A tool choice to use for the chat completion.
* **`temperature`**
  (`[float](https://docs.python.org/3/library/functions.html#float)`, default:
  `0.2`
  )
  –

  The temperature to use for sampling.
* **`top_p`**
  (`[float](https://docs.python.org/3/library/functions.html#float)`, default:
  `0.95`
  )
  –

  The top-p value to use for nucleus sampling. Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" <https://arxiv.org/abs/1904.09751>
* **`top_k`**
  (`[int](https://docs.python.org/3/library/functions.html#int)`, default:
  `40`
  )
  –

  The top-k value to use for sampling. Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" <https://arxiv.org/abs/1904.09751>
* **`min_p`**
  (`[float](https://docs.python.org/3/library/functions.html#float)`, default:
  `0.05`
  )
  –

  The min-p value to use for minimum p sampling. Minimum P sampling as described in <https://github.com/ggerganov/llama.cpp/pull/3841>
* **`typical_p`**
  (`[float](https://docs.python.org/3/library/functions.html#float)`, default:
  `1.0`
  )
  –

  The typical-p value to use for sampling. Locally Typical Sampling implementation described in the paper <https://arxiv.org/abs/2202.00666>.
* **`stream`**
  (`[bool](https://docs.python.org/3/library/functions.html#bool)`, default:
  `False`
  )
  –

  Whether to stream the results.
* **`stop`**
  (`[Optional](https://docs.python.org/3/library/typing.html#typing.Optional "llama_cpp.llama_types.Optional")[[Union](https://docs.python.org/3/library/typing.html#typing.Union "llama_cpp.llama_types.Union")[[str](https://docs.python.org/3/library/stdtypes.html#str), [List](https://docs.python.org/3/library/typing.html#typing.List "llama_cpp.llama_types.List")[[str](https://docs.python.org/3/library/stdtypes.html#str)]]]`, default:
  `[]`
  )
  –

  A list of strings to stop generation when encountered.
* **`seed`**
  (`[Optional](https://docs.python.org/3/library/typing.html#typing.Optional "llama_cpp.llama_types.Optional")[[int](https://docs.python.org/3/library/functions.html#int)]`, default:
  `None`
  )
  –

  The seed to use for sampling.
* **`response_format`**
  (`[Optional](https://docs.python.org/3/library/typing.html#typing.Optional "llama_cpp.llama_types.Optional")[[ChatCompletionRequestResponseFormat](#llama_cpp.llama_types.ChatCompletionRequestResponseFormat "ChatCompletionRequestResponseFormat (llama_cpp.llama_types.ChatCompletionRequestResponseFormat)")]`, default:
  `None`
  )
  –

  The response format to use for the chat completion. Use { "type": "json\_object" } to contstrain output to only valid json.
* **`max_tokens`**
  (`[Optional](https://docs.python.org/3/library/typing.html#typing.Optional "llama_cpp.llama_types.Optional")[[int](https://docs.python.org/3/library/functions.html#int)]`, default:
  `None`
  )
  –

  The maximum number of tokens to generate. If max\_tokens <= 0 or None, the maximum number of tokens to generate is unlimited and depends on n\_ctx.
* **`presence_penalty`**
  (`[float](https://docs.python.org/3/library/functions.html#float)`, default:
  `0.0`
  )
  –

  The penalty to apply to tokens based on their presence in the prompt.
* **`frequency_penalty`**
  (`[float](https://docs.python.org/3/library/functions.html#float)`, default:
  `0.0`
  )
  –

  The penalty to apply to tokens based on their frequency in the prompt.
* **`repeat_penalty`**
  (`[float](https://docs.python.org/3/library/functions.html#float)`, default:
  `1.0`
  )
  –

  The penalty to apply to repeated tokens.
* **`tfs_z`**
  (`[float](https://docs.python.org/3/library/functions.html#float)`, default:
  `1.0`
  )
  –

  The tail-free sampling parameter.
* **`mirostat_mode`**
  (`[int](https://docs.python.org/3/library/functions.html#int)`, default:
  `0`
  )
  –

  The mirostat sampling mode.
* **`mirostat_tau`**
  (`[float](https://docs.python.org/3/library/functions.html#float)`, default:
  `5.0`
  )
  –

  The mirostat sampling tau parameter.
* **`mirostat_eta`**
  (`[float](https://docs.python.org/3/library/functions.html#float)`, default:
  `0.1`
  )
  –

  The mirostat sampling eta parameter.
* **`model`**
  (`[Optional](https://docs.python.org/3/library/typing.html#typing.Optional "llama_cpp.llama_types.Optional")[[str](https://docs.python.org/3/library/stdtypes.html#str)]`, default:
  `None`
  )
  –

  The name to use for the model in the completion object.
* **`logits_processor`**
  (`[Optional](https://docs.python.org/3/library/typing.html#typing.Optional "llama_cpp.llama_types.Optional")[[LogitsProcessorList](#llama_cpp.LogitsProcessorList "llama_cpp.LogitsProcessorList (llama_cpp.llama.LogitsProcessorList)")]`, default:
  `None`
  )
  –

  A list of logits processors to use.
* **`grammar`**
  (`[Optional](https://docs.python.org/3/library/typing.html#typing.Optional "llama_cpp.llama_types.Optional")[[LlamaGrammar](#llama_cpp.LlamaGrammar "llama_cpp.LlamaGrammar (llama_cpp.llama_grammar.LlamaGrammar)")]`, default:
  `None`
  )
  –

  A grammar to use.
* **`logit_bias`**
  (`[Optional](https://docs.python.org/3/library/typing.html#typing.Optional "llama_cpp.llama_types.Optional")[[Dict](https://docs.python.org/3/library/typing.html#typing.Dict "llama_cpp.llama_types.Dict")[[int](https://docs.python.org/3/library/functions.html#int), [float](https://docs.python.org/3/library/functions.html#float)]]`, default:
  `None`
  )
  –

  A logit bias to use.

Returns:

* `[Union](https://docs.python.org/3/library/typing.html#typing.Union "llama_cpp.llama_types.Union")[[CreateChatCompletionResponse](#llama_cpp.llama_types.CreateChatCompletionResponse "CreateChatCompletionResponse (llama_cpp.llama_types.CreateChatCompletionResponse)"), [Iterator](https://docs.python.org/3/library/typing.html#typing.Iterator "typing.Iterator")[[CreateChatCompletionStreamResponse](#llama_cpp.llama_types.CreateChatCompletionStreamResponse "CreateChatCompletionStreamResponse (llama_cpp.llama_types.CreateChatCompletionStreamResponse)")]]`
  –

  Generated chat completion or a stream of chat completion chunks.

#### `create_chat_completion_openai_v1(*args, **kwargs)`

Generate a chat completion with return type based on the the OpenAI v1 API.

OpenAI python package is required to use this method.

You can install it with `pip install openai`.

Parameters:

* **`*args`**
  (`[Any](https://docs.python.org/3/library/typing.html#typing.Any "llama_cpp.llama_types.Any")`, default:
  `()`
  )
  –

  Positional arguments to pass to create\_chat\_completion.
* **`**kwargs`**
  (`[Any](https://docs.python.org/3/library/typing.html#typing.Any "llama_cpp.llama_types.Any")`, default:
  `{}`
  )
  –

  Keyword arguments to pass to create\_chat\_completion.

Returns:

* –

  Generated chat completion or a stream of chat completion chunks.

#### `set_cache(cache)`

Set the cache.

Parameters:

* **`cache`**
  (`[Optional](https://docs.python.org/3/library/typing.html#typing.Optional "llama_cpp.llama_types.Optional")[BaseLlamaCache]`)
  –

  The cache to set.

#### `save_state()`

#### `load_state(state)`

#### `token_bos()`

Return the beginning-of-sequence token.

#### `token_eos()`

Return the end-of-sequence token.

#### `from_pretrained(repo_id, filename, additional_files=None, local_dir=None, local_dir_use_symlinks='auto', cache_dir=None, **kwargs)` `classmethod`

Create a Llama model from a pretrained model name or path.
This method requires the huggingface-hub package.
You can install it with `pip install huggingface-hub`.

Parameters:

* **`repo_id`**
  (`[str](https://docs.python.org/3/library/stdtypes.html#str)`)
  –

  The model repo id.
* **`filename`**
  (`[Optional](https://docs.python.org/3/library/typing.html#typing.Optional "llama_cpp.llama_types.Optional")[[str](https://docs.python.org/3/library/stdtypes.html#str)]`)
  –

  A filename or glob pattern to match the model file in the repo.
* **`additional_files`**
  (`[Optional](https://docs.python.org/3/library/typing.html#typing.Optional "llama_cpp.llama_types.Optional")[[List](https://docs.python.org/3/library/typing.html#typing.List "llama_cpp.llama_types.List")]`, default:
  `None`
  )
  –

  A list of filenames or glob patterns to match additional model files in the repo.
* **`local_dir`**
  (`[Optional](https://docs.python.org/3/library/typing.html#typing.Optional "llama_cpp.llama_types.Optional")[[Union](https://docs.python.org/3/library/typing.html#typing.Union "llama_cpp.llama_types.Union")[[str](https://docs.python.org/3/library/stdtypes.html#str), [PathLike](https://docs.python.org/3/library/os.html#os.PathLike "os.PathLike")[[str](https://docs.python.org/3/library/stdtypes.html#str)]]]`, default:
  `None`
  )
  –

  The local directory to save the model to.
* **`local_dir_use_symlinks`**
  (`[Union](https://docs.python.org/3/library/typing.html#typing.Union "llama_cpp.llama_types.Union")[[bool](https://docs.python.org/3/library/functions.html#bool), Literal['auto']]`, default:
  `'auto'`
  )
  –

  Whether to use symlinks when downloading the model.
* **`**kwargs`**
  (`[Any](https://docs.python.org/3/library/typing.html#typing.Any "llama_cpp.llama_types.Any")`, default:
  `{}`
  )
  –

  Additional keyword arguments to pass to the Llama constructor.

Returns:

* `'Llama'`
  –

  A Llama model.

### `llama_cpp.LlamaGrammar`

#### `from_string(grammar, verbose=True)` `classmethod`

#### `from_json_schema(json_schema, verbose=True)` `classmethod`

### `llama_cpp.LlamaCache = LlamaRAMCache` `module-attribute`

### `llama_cpp.LlamaState`

### `llama_cpp.LogitsProcessor = Callable[[npt.NDArray[np.intc], npt.NDArray[np.single]], npt.NDArray[np.single]]` `module-attribute`

### `llama_cpp.LogitsProcessorList`

Bases: `[List](https://docs.python.org/3/library/typing.html#typing.List "llama_cpp.llama_types.List")[[LogitsProcessor](#llama_cpp.LogitsProcessor "llama_cpp.LogitsProcessor = Callable[[npt.NDArray[np.intc], npt.NDArray[np.single]], npt.NDArray[np.single]]

      module-attribute
   (llama_cpp.llama.LogitsProcessor)")]`

### `llama_cpp.StoppingCriteria = Callable[[npt.NDArray[np.intc], npt.NDArray[np.single]], bool]` `module-attribute`

### `llama_cpp.StoppingCriteriaList`

Bases: `[List](https://docs.python.org/3/library/typing.html#typing.List "llama_cpp.llama_types.List")[[StoppingCriteria](#llama_cpp.StoppingCriteria "llama_cpp.StoppingCriteria = Callable[[npt.NDArray[np.intc], npt.NDArray[np.single]], bool]

      module-attribute
   (llama_cpp.llama.StoppingCriteria)")]`

## Low Level API

Low-level Python bindings for llama.cpp using Python's ctypes library.

### `llama_cpp.llama_cpp`

#### `llama_vocab_p = NewType('llama_vocab_p', int)` `module-attribute`

#### `llama_vocab_p_ctypes = ctypes.c_void_p` `module-attribute`

#### `llama_model_p = NewType('llama_model_p', int)` `module-attribute`

#### `llama_model_p_ctypes = ctypes.c_void_p` `module-attribute`

#### `llama_context_p = NewType('llama_context_p', int)` `module-attribute`

#### `llama_context_p_ctypes = ctypes.c_void_p` `module-attribute`

#### `llama_kv_cache_p = NewType('llama_kv_cache_p', int)` `module-attribute`

#### `llama_kv_cache_p_ctypes = ctypes.c_void_p` `module-attribute`

#### `llama_pos = ctypes.c_int32` `module-attribute`

#### `llama_token = ctypes.c_int32` `module-attribute`

#### `llama_token_p = ctypes.POINTER(llama_token)` `module-attribute`

#### `llama_seq_id = ctypes.c_int32` `module-attribute`

#### `llama_token_data`

Bases: `[Structure](https://docs.python.org/3/library/ctypes.html#ctypes.Structure "ctypes.Structure")`

Used to store token data

Attributes:

* **`id`**
  (`[llama_token](#llama_cpp.llama_cpp.llama_token "llama_token = ctypes.c_int32

        module-attribute
     (llama_cpp.llama_cpp.llama_token)")`)
  –

  token id
* **`logit`**
  (`[float](https://docs.python.org/3/library/functions.html#float)`)
  –

  log-odds of the token
* **`p`**
  (`[float](https://docs.python.org/3/library/functions.html#float)`)
  –

  probability of the token

#### `llama_token_data_p = ctypes.POINTER(llama_token_data)` `module-attribute`

#### `llama_token_data_array`

Bases: `[Structure](https://docs.python.org/3/library/ctypes.html#ctypes.Structure "ctypes.Structure")`

Used to sample tokens given logits

Attributes:

* **`data`**
  (`[Array](https://docs.python.org/3/library/ctypes.html#ctypes.Array "ctypes.Array")[[llama_token_data](#llama_cpp.llama_cpp.llama_token_data "llama_token_data (llama_cpp.llama_cpp.llama_token_data)")]`)
  –

  token data
* **`size`**
  (`[int](https://docs.python.org/3/library/functions.html#int)`)
  –

  size of the array
* **`selected`**
  (`[int](https://docs.python.org/3/library/functions.html#int)`)
  –

  index in the data array (i.e. not the token id)
* **`sorted`**
  (`[bool](https://docs.python.org/3/library/functions.html#bool)`)
  –

  whether the array is sorted

#### `llama_token_data_array_p = ctypes.POINTER(llama_token_data_array)` `module-attribute`

#### `llama_progress_callback = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_float, ctypes.c_void_p)` `module-attribute`

#### `llama_batch`

Bases: `[Structure](https://docs.python.org/3/library/ctypes.html#ctypes.Structure "ctypes.Structure")`

Input data for llama\_decode

A llama\_batch object can contain input about one or many sequences

The provided arrays (i.e. token, embd, pos, etc.) must have size of n\_tokens

Attributes:

* **`n_tokens`**
  (`[int](https://docs.python.org/3/library/functions.html#int)`)
  –

  number of tokens
* **`token`**
  (`[Array](https://docs.python.org/3/library/ctypes.html#ctypes.Array "ctypes.Array")[[llama_token](#llama_cpp.llama_cpp.llama_token "llama_token = ctypes.c_int32

        module-attribute
     (llama_cpp.llama_cpp.llama_token)")]`)
  –

  the token ids of the input (used when embd is NULL)
* **`embd`**
  (`[Array](https://docs.python.org/3/library/ctypes.html#ctypes.Array "ctypes.Array")[c_float]`)
  –

  token embeddings (i.e. float vector of size n\_embd) (used when token is NULL)
* **`pos`**
  (`[Array](https://docs.python.org/3/library/ctypes.html#ctypes.Array "ctypes.Array")[[Array](https://docs.python.org/3/library/ctypes.html#ctypes.Array "ctypes.Array")[[llama_pos](#llama_cpp.llama_cpp.llama_pos "llama_pos = ctypes.c_int32

        module-attribute
     (llama_cpp.llama_cpp.llama_pos)")]]`)
  –

  the positions of the respective token in the sequence
* **`seq_id`**
  (`[Array](https://docs.python.org/3/library/ctypes.html#ctypes.Array "ctypes.Array")[[Array](https://docs.python.org/3/library/ctypes.html#ctypes.Array "ctypes.Array")[[llama_seq_id](#llama_cpp.llama_cpp.llama_seq_id "llama_seq_id = ctypes.c_int32

        module-attribute
     (llama_cpp.llama_cpp.llama_seq_id)")]]`)
  –

  the sequence to which the respective token belongs
* **`logits`**
  (`[Array](https://docs.python.org/3/library/ctypes.html#ctypes.Array "ctypes.Array")[c_int8]`)
  –

  if zero, the logits for the respective token will not be output

#### `llama_model_kv_override_value`

Bases: `[Union](https://docs.python.org/3/library/ctypes.html#ctypes.Union "ctypes.Union")`

#### `llama_model_kv_override`

Bases: `[Structure](https://docs.python.org/3/library/ctypes.html#ctypes.Structure "ctypes.Structure")`

#### `llama_model_params`

Bases: `[Structure](https://docs.python.org/3/library/ctypes.html#ctypes.Structure "ctypes.Structure")`

Parameters for llama\_model

Attributes:

* **`devices`**
  (`[Array](https://docs.python.org/3/library/ctypes.html#ctypes.Array "ctypes.Array")[ggml_backend_dev_t]`)
  –

  NULL-terminated list of devices to use for offloading (if NULL, all available devices are used)
* **`tensor_buft_overrides`**
  (`[Array](https://docs.python.org/3/library/ctypes.html#ctypes.Array "ctypes.Array")[llama_model_tensor_buft_override]`)
  –

  NULL-terminated list of buffer types to use for tensors that match a pattern
* **`n_gpu_layers`**
  (`[int](https://docs.python.org/3/library/functions.html#int)`)
  –

  number of layers to store in VRAM
* **`split_mode`**
  (`[int](https://docs.python.org/3/library/functions.html#int)`)
  –

  how to split the model across multiple GPUs
* **`main_gpu`**
  (`[int](https://docs.python.org/3/library/functions.html#int)`)
  –

  the GPU that is used for the entire model. main\_gpu interpretation depends on split\_mode: LLAMA\_SPLIT\_NONE: the GPU that is used for the entire model LLAMA\_SPLIT\_ROW: the GPU that is used for small tensors and intermediate results LLAMA\_SPLIT\_LAYER: ignored
* **`tensor_split`**
  (`[Array](https://docs.python.org/3/library/ctypes.html#ctypes.Array "ctypes.Array")[c_float]`)
  –

  proportion of the model (layers or rows) to offload to each GPU, size: llama\_max\_devices()
* **`progress_callback`**
  (`[llama_progress_callback](#llama_cpp.llama_cpp.llama_progress_callback "llama_progress_callback = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_float, ctypes.c_void_p)

        module-attribute
     (llama_cpp.llama_cpp.llama_progress_callback)")`)
  –

  called with a progress value between 0.0 and 1.0. Pass NULL to disable. If the provided progress\_callback returns true, model loading continues. If it returns false, model loading is immediately aborted.
* **`progress_callback_user_data`**
  (`c_void_p`)
  –

  context pointer passed to the progress callback
* **`kv_overrides`**
  (`[Array](https://docs.python.org/3/library/ctypes.html#ctypes.Array "ctypes.Array")[[llama_model_kv_override](#llama_cpp.llama_cpp.llama_model_kv_override "llama_model_kv_override (llama_cpp.llama_cpp.llama_model_kv_override)")]`)
  –

  override key-value pairs of the model meta data
* **`vocab_only`**
  (`[bool](https://docs.python.org/3/library/functions.html#bool)`)
  –

  only load the vocabulary, no weights
* **`use_mmap`**
  (`[bool](https://docs.python.org/3/library/functions.html#bool)`)
  –

  use mmap if possible
* **`use_mlock`**
  (`[bool](https://docs.python.org/3/library/functions.html#bool)`)
  –

  force system to keep model in RAM
* **`check_tensors`**
  (`[bool](https://docs.python.org/3/library/functions.html#bool)`)
  –

  validate model tensor data

#### `llama_context_params`

Bases: `[Structure](https://docs.python.org/3/library/ctypes.html#ctypes.Structure "ctypes.Structure")`

Parameters for llama\_context

Attributes:

* **`n_ctx`**
  (`[int](https://docs.python.org/3/library/functions.html#int)`)
  –

  text context, 0 = from model
* **`n_batch`**
  (`[int](https://docs.python.org/3/library/functions.html#int)`)
  –

  logical maximum batch size that can be submitted to llama\_decode
* **`n_ubatch`**
  (`[int](https://docs.python.org/3/library/functions.html#int)`)
  –

  physical maximum batch size
* **`n_seq_max`**
  (`[int](https://docs.python.org/3/library/functions.html#int)`)
  –

  max number of sequences (i.e. distinct states for recurrent models)
* **`n_threads`**
  (`[int](https://docs.python.org/3/library/functions.html#int)`)
  –

  number of threads to use for generation
* **`n_threads_batch`**
  (`[int](https://docs.python.org/3/library/functions.html#int)`)
  –

  number of threads to use for batch processing
* **`rope_scaling_type`**
  (`[int](https://docs.python.org/3/library/functions.html#int)`)
  –

  RoPE scaling type, from `enum llama_rope_scaling_type`
* **`pooling_type`**
  (`[int](https://docs.python.org/3/library/functions.html#int)`)
  –

  whether to pool (sum) embedding results by sequence id (ignored if no pooling layer)
* **`attention_type`**
  (`[int](https://docs.python.org/3/library/functions.html#int)`)
  –

  attention type to use for embeddings
* **`rope_freq_base`**
  (`[float](https://docs.python.org/3/library/functions.html#float)`)
  –

  RoPE base frequency, 0 = from model
* **`rope_freq_scale`**
  (`[float](https://docs.python.org/3/library/functions.html#float)`)
  –

  RoPE frequency scaling factor, 0 = from model
* **`yarn_ext_factor`**
  (`[float](https://docs.python.org/3/library/functions.html#float)`)
  –

  YaRN extrapolation mix factor, negative = from model
* **`yarn_attn_factor`**
  (`[float](https://docs.python.org/3/library/functions.html#float)`)
  –

  YaRN magnitude scaling factor
* **`yarn_beta_fast`**
  (`[float](https://docs.python.org/3/library/functions.html#float)`)
  –

  YaRN low correction dim
* **`yarn_beta_slow`**
  (`[float](https://docs.python.org/3/library/functions.html#float)`)
  –

  YaRN high correction dim
* **`yarn_orig_ctx`**
  (`[int](https://docs.python.org/3/library/functions.html#int)`)
  –

  YaRN original context size
* **`defrag_thold`**
  (`[float](https://docs.python.org/3/library/functions.html#float)`)
  –

  defragment the KV cache if holes/size > thold, < 0 disabled (default)
* **`cb_eval`**
  (`ggml_backend_sched_eval_callback`)
  –

  callback for scheduling eval
* **`cb_eval_user_data`**
  (`c_void_p`)
  –

  user data for cb\_eval
* **`type_k`**
  (`[int](https://docs.python.org/3/library/functions.html#int)`)
  –

  data type for K cache
* **`type_v`**
  (`[int](https://docs.python.org/3/library/functions.html#int)`)
  –

  data type for V cache
* **`logits_all`**
  (`[bool](https://docs.python.org/3/library/functions.html#bool)`)
  –

  the llama\_decode() call computes all logits, not just the last one (DEPRECATED - set llama\_batch.logits instead)
* **`embeddings`**
  (`[bool](https://docs.python.org/3/library/functions.html#bool)`)
  –

  if true, extract embeddings (together with logits)
* **`offload_kqv`**
  (`[bool](https://docs.python.org/3/library/functions.html#bool)`)
  –

  whether to offload the KQV ops (including the KV cache) to GPU
* **`flash_attn`**
  (`[bool](https://docs.python.org/3/library/functions.html#bool)`)
  –

  whether to use flash attention
* **`no_perf`**
  (`[bool](https://docs.python.org/3/library/functions.html#bool)`)
  –

  whether to measure performance timings
* **`abort_callback`**
  (`ggml_abort_callback`)
  –

  abort callback if it returns true, execution of llama\_decode() will be aborted
* **`abort_callback_data`**
  (`c_void_p`)
  –

  data for abort\_callback

#### `llama_log_callback = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)` `module-attribute`

Signature for logging events
Note that text includes the new line character at the end for most events.
If your logging mechanism cannot handle that, check if the last character is '
' and strip it
if it exists.
It might not exist for progress report where '.' is output repeatedly.

#### `llama_model_quantize_params`

Bases: `[Structure](https://docs.python.org/3/library/ctypes.html#ctypes.Structure "ctypes.Structure")`

Parameters for llama\_model\_quantize

Attributes:

* **`nthread`**
  (`[int](https://docs.python.org/3/library/functions.html#int)`)
  –

  number of threads to use for quantizing, if <=0 will use std:![🧵](https://cdn.jsdelivr.net/gh/jdecked/twemoji@15.1.0/assets/svg/1f9f5.svg ":thread:"):hardware\_concurrency()
* **`ftype`**
  (`[int](https://docs.python.org/3/library/functions.html#int)`)
  –

  quantize to this llama\_ftype
* **`output_tensor_type`**
  (`[int](https://docs.python.org/3/library/functions.html#int)`)
  –

  output tensor type
* **`token_embedding_type`**
  (`[int](https://docs.python.org/3/library/functions.html#int)`)
  –

  token embeddings tensor type
* **`allow_requantize`**
  (`[bool](https://docs.python.org/3/library/functions.html#bool)`)
  –

  allow quantizing non-f32/f16 tensors
* **`quantize_output_tensor`**
  (`[bool](https://docs.python.org/3/library/functions.html#bool)`)
  –

  quantize output.weight
* **`only_copy`**
  (`[bool](https://docs.python.org/3/library/functions.html#bool)`)
  –

  only copy tensors - ftype, allow\_requantize and quantize\_output\_tensor are ignored
* **`pure`**
  (`[bool](https://docs.python.org/3/library/functions.html#bool)`)
  –

  quantize all tensors to the default type
* **`keep_split`**
  (`[bool](https://docs.python.org/3/library/functions.html#bool)`)
  –

  quantize to the same number of shards
* **`imatrix`**
  (`[c_void_p](https://docs.python.org/3/library/ctypes.html#ctypes.c_void_p "ctypes.c_void_p")`)
  –

  pointer to importance matrix data
* **`kv_overrides`**
  (`[c_void_p](https://docs.python.org/3/library/ctypes.html#ctypes.c_void_p "ctypes.c_void_p")`)
  –

  pointer to vector containing overrides
* **`tensor_types`**
  (`[c_void_p](https://docs.python.org/3/library/ctypes.html#ctypes.c_void_p "ctypes.c_void_p")`)
  –

  pointer to vector containing tensor types

#### `llama_logit_bias`

Bases: `[Structure](https://docs.python.org/3/library/ctypes.html#ctypes.Structure "ctypes.Structure")`

Used to store logit bias

Attributes:

* **`token`**
  (`[llama_token](#llama_cpp.llama_cpp.llama_token "llama_token = ctypes.c_int32

        module-attribute
     (llama_cpp.llama_cpp.llama_token)")`)
  –

  token id
* **`bias`**
  (`[float](https://docs.python.org/3/library/functions.html#float)`)
  –

  bias

#### `llama_logit_bias_p = ctypes.POINTER(llama_logit_bias)` `module-attribute`

#### `llama_sampler_chain_params`

Bases: `[Structure](https://docs.python.org/3/library/ctypes.html#ctypes.Structure "ctypes.Structure")`

Parameters for llama\_sampler\_chain

Attributes:

* **`no_perf`**
  (`[bool](https://docs.python.org/3/library/functions.html#bool)`)
  –

  whether to measure performance timings

#### `llama_chat_message`

Bases: `[Structure](https://docs.python.org/3/library/ctypes.html#ctypes.Structure "ctypes.Structure")`

#### `llama_adapter_lora_p = ctypes.c_void_p` `module-attribute`

#### `llama_adapter_lora_p_ctypes = ctypes.POINTER(ctypes.c_void_p)` `module-attribute`

#### `llama_model_default_params()`

Get default parameters for llama\_model

#### `llama_context_default_params()`

Get default parameters for llama\_context

#### `llama_sampler_chain_default_params()`

Get default parameters for llama\_sampler\_chain

#### `llama_model_quantize_default_params()`

Get default parameters for llama\_model\_quantize

#### `llama_backend_init()`

Initialize the llama + ggml backend
If numa is true, use NUMA optimizations
Call once at the start of the program

#### `llama_backend_free()`

Call once at the end of the program - currently only used for MPI

#### `llama_numa_init(numa)`

#### `llama_load_model_from_file(path_model, params)`

#### `llama_model_load_from_file(path_model, params)`

Load the model from a file

If the file is split into multiple parts, the file name must follow this pattern: -%05d-of-%05d.gguf

If the split file name does not follow this pattern, use llama\_model\_load\_from\_splits

#### `llama_model_load_from_splits(paths, n_paths, params)`

Load the model from multiple splits (support custom naming scheme)

The paths must be in the correct order

#### `llama_free_model(model)`

#### `llama_model_free(model)`

#### `llama_init_from_model(model, params)`

#### `llama_new_context_with_model(model, params)`

#### `llama_free(ctx)`

Frees all allocated memory

#### `llama_time_us()`

#### `llama_max_devices()`

#### `llama_supports_mmap()`

#### `llama_supports_mlock()`

#### `llama_supports_gpu_offload()`

#### `llama_supports_rpc()`

#### `llama_n_ctx(ctx)`

#### `llama_n_batch(ctx)`

#### `llama_n_ubatch(ctx)`

#### `llama_n_seq_max(ctx)`

#### `llama_n_ctx_train(model)`

#### `llama_n_embd(model)`

#### `llama_n_layer(model)`

#### `llama_n_head(model)`

#### `llama_n_vocab(model)`

#### `llama_get_model(ctx)`

#### `llama_get_kv_self(ctx)`

Get the KV cache for self-attention

#### `llama_pooling_type(ctx)`

#### `llama_model_get_vocab(model)`

#### `llama_model_rope_type(model)`

#### `llama_model_n_ctx_train(model)`

#### `llama_model_n_embd(model)`

#### `llama_model_n_layer(model)`

#### `llama_model_n_head(model)`

#### `llama_model_n_head_kv(model)`

#### `llama_model_rope_freq_scale_train(model)`

#### `llama_vocab_type(model)`

#### `llama_vocab_n_tokens(vocab)`

#### `llama_model_meta_val_str(model, key, buf, buf_size)`

Get metadata value as a string by key name

#### `llama_model_meta_count(model)`

Get the number of metadata key/value pairs

#### `llama_model_meta_key_by_index(model, i, buf, buf_size)`

Get metadata key name by index

#### `llama_model_meta_val_str_by_index(model, i, buf, buf_size)`

Get metadata value as a string by index

#### `llama_model_desc(model, buf, buf_size)`

Get a string describing the model type

#### `llama_model_size(model)`

Returns the total size of all the tensors in the model in bytes

#### `llama_model_chat_template(model, name)`

Get the default chat template. Returns None if not available
If name is None, returns the default chat template

#### `llama_model_n_params(model)`

Returns the total number of parameters in the model

#### `llama_model_has_encoder(model)`

Returns true if the model contains an encoder that requires llama\_encode() call

#### `llama_model_has_decoder(model)`

Returns true if the model contains a decoder that requires llama\_decode() call

#### `llama_model_decoder_start_token(model)`

For encoder-decoder models, this function returns id of the token that must be provided
to the decoder to start generating output sequence. For other models, it returns -1.

#### `llama_model_is_recurrent(model)`

Returns true if the model is recurrent (like Mamba, RWKV, etc.)

#### `llama_model_quantize(fname_inp, fname_out, params)`

Returns 0 on success

#### `llama_adapter_lora_init(model, path_lora)`

#### `llama_adapter_lora_free(adapter)`

#### `llama_set_adapter_lora(ctx, adapter, scale)`

Add a loaded LoRA adapter to given context
This will not modify model's weight

#### `llama_rm_adapter_lora(ctx, adapter)`

Remove a specific LoRA adapter from given context
Return -1 if the adapter is not present in the context

#### `llama_clear_adapter_lora(ctx)`

Remove all LoRA adapters from given context

#### `llama_apply_adapter_cvec(ctx, data, len, n_embd, il_start, il_end)`

Apply a loaded control vector to a llama\_context, or if data is NULL, clear
the currently loaded vector.
n\_embd should be the size of a single layer's control, and data should point
to an n\_embd x n\_layers buffer starting from layer 1.
il\_start and il\_end are the layer range the vector should apply to (both inclusive)
See llama\_control\_vector\_load in common to load a control vector.

#### `llama_kv_cache_view_cell`

Bases: `[Structure](https://docs.python.org/3/library/ctypes.html#ctypes.Structure "ctypes.Structure")`

Information associated with an individual cell in the KV cache view.

Attributes:

* **`pos`**
  (`[llama_pos](#llama_cpp.llama_cpp.llama_pos "llama_pos = ctypes.c_int32

        module-attribute
     (llama_cpp.llama_cpp.llama_pos)")`)
  –

  The position for this cell. Takes KV cache shifts into account.
  May be negative if the cell is not populated.

#### `llama_kv_cache_view`

Bases: `[Structure](https://docs.python.org/3/library/ctypes.html#ctypes.Structure "ctypes.Structure")`

#### `llama_kv_cache_view_p = ctypes.POINTER(llama_kv_cache_view)` `module-attribute`

#### `llama_kv_cache_view_init(ctx, n_seq_max)`

Create an empty KV cache view. (use only for debugging purposes)

#### `llama_kv_cache_view_free(view)`

Free a KV cache view. (use only for debugging purposes)

#### `llama_kv_cache_view_update(ctx, view)`

Update the KV cache view structure with the current state of the KV cache. (use only for debugging purposes)

#### `llama_kv_self_n_tokens(ctx)`

Returns the number of tokens in the KV cache (slow, use only for debug)
If a KV cell has multiple sequences assigned to it, it will be counted multiple times

#### `llama_get_kv_cache_token_count(ctx)`

Returns the number of tokens in the KV cache (slow, use only for debug)
If a KV cell has multiple sequences assigned to it, it will be counted multiple times

#### `llama_kv_self_used_cells(ctx)`

Returns the number of used KV cells (i.e. have at least one sequence assigned to them)

#### `llama_get_kv_cache_used_cells(ctx)`

Returns the number of used KV cells (i.e. have at least one sequence assigned to them)

#### `llama_kv_self_clear(ctx)`

Clear the KV cache - both cell info is erased and KV data is zeroed

#### `llama_kv_cache_clear(ctx)`

Clear the KV cache

#### `llama_kv_cache_seq_rm(ctx, seq_id, p0, p1)`

Removes all tokens that belong to the specified sequence and have positions in [p0, p1)

Returns false if a partial sequence cannot be removed. Removing a whole sequence never fails

seq\_id < 0 : match any sequence
p0 < 0 : [0, p1]
p1 < 0 : [p0, inf)

#### `llama_kv_self_seq_cp(ctx, seq_id_src, seq_id_dst, p0, p1)`

Copy all tokens that belong to the specified sequence to another sequence
Note that this does not allocate extra KV cache memory - it simply assigns the tokens to the new sequence
p0 < 0 : [0, p1]
p1 < 0 : [p0, inf)

#### `llama_kv_cache_seq_cp(ctx, seq_id_src, seq_id_dst, p0, p1)`

Copy all tokens that belong to the specified sequence to another sequence
Note that this does not allocate extra KV cache memory - it simply assigns the tokens to the new sequence
p0 < 0 : [0, p1]
p1 < 0 : [p0, inf)

#### `llama_kv_self_seq_keep(ctx, seq_id)`

Removes all tokens that do not belong to the specified sequence

#### `llama_kv_cache_seq_keep(ctx, seq_id)`

Removes all tokens that do not belong to the specified sequence

#### `llama_kv_self_seq_add(ctx, seq_id, p0, p1, delta)`

Adds relative position "delta" to all tokens that belong to the specified sequence and have positions in [p0, p1)
If the KV cache is RoPEd, the KV data is updated accordingly:
- lazily on next llama\_decode()
- explicitly with llama\_kv\_cache\_update()
p0 < 0 : [0, p1]
p1 < 0 : [p0, inf)

#### `llama_kv_cache_seq_add(ctx, seq_id, p0, p1, delta)`

Adds relative position "delta" to all tokens that belong to the specified sequence and have positions in [p0, p1)
If the KV cache is RoPEd, the KV data is updated accordingly:
- lazily on next llama\_decode()
- explicitly with llama\_kv\_cache\_update()
p0 < 0 : [0, p1]
p1 < 0 : [p0, inf)

#### `llama_kv_self_seq_div(ctx, seq_id, p0, p1, d)`

Integer division of the positions by factor of `d > 1`
If the KV cache is RoPEd, the KV data is updated accordingly
p0 < 0 : [0, p1]
p1 < 0 : [p0, inf)

#### `llama_kv_cache_seq_div(ctx, seq_id, p0, p1, d)`

Integer division of the positions by factor of `d > 1`
If the KV cache is RoPEd, the KV data is updated accordingly
p0 < 0 : [0, p1]
p1 < 0 : [p0, inf)

#### `llama_kv_self_seq_pos_max(ctx, seq_id)`

Returns the largest position present in the KV cache for the specified sequence

#### `llama_kv_self_defrag(ctx)`

Defragment the KV cache
This will be applied:
- lazily on next llama\_decode()
- explicitly with llama\_kv\_cache\_update()

#### `llama_kv_cache_defrag(ctx)`

Defragment the KV cache
This will be applied:
- lazily on next llama\_decode()
- explicitly with llama\_kv\_cache\_update()

#### `llama_kv_self_update(ctx)`

Apply the KV cache updates (such as K-shifts, defragmentation, etc.)

#### `llama_kv_cache_update(ctx)`

Apply the KV cache updates (such as K-shifts, defragmentation, etc.)

#### `llama_kv_self_can_shift(ctx)`

Check if the context supports KV cache shifting

#### `llama_kv_cache_can_shift(ctx)`

Check if the context supports KV cache shifting

#### `llama_state_get_size(ctx)`

Returns the *actual* size in bytes of the state (rng, logits, embedding and kv\_cache) - will often be smaller after compacting tokens

#### `llama_get_state_size(ctx)`

Returns the maximum size in bytes of the state (rng, logits, embedding
and kv\_cache) - will often be smaller after compacting tokens

#### `llama_state_get_data(ctx, dst, size)`

Copies the state to the specified destination address.
Destination needs to have allocated enough memory.
Returns the number of bytes copied

#### `llama_copy_state_data(ctx, dst)`

Copies the state to the specified destination address.
Destination needs to have allocated enough memory.
Returns the number of bytes copied

#### `llama_state_set_data(ctx, src, size)`

Set the state reading from the specified address
Returns the number of bytes read

#### `llama_set_state_data(ctx, src)`

Set the state reading from the specified address

#### `llama_state_load_file(ctx, path_session, tokens_out, n_token_capacity, n_token_count_out)`

#### `llama_load_session_file(ctx, path_session, tokens_out, n_token_capacity, n_token_count_out)`

#### `llama_state_save_file(ctx, path_session, tokens, n_token_count)`

#### `llama_save_session_file(ctx, path_session, tokens, n_token_count)`

#### `llama_state_seq_get_size(ctx, seq_id)`

Get the exact size needed to copy the KV cache of a single sequence

#### `llama_state_seq_get_data(ctx, dst, size, seq_id)`

Copy the KV cache of a single sequence into the specified buffer

#### `llama_state_seq_set_data(ctx, src, size, dest_seq_id)`

Copy the sequence data (originally copied with `llama_state_seq_get_data`) into the specified sequence

#### `llama_state_seq_save_file(ctx, filepath, seq_id, tokens, n_token_count)`

#### `llama_state_seq_load_file(ctx, filepath, dest_seq_id, tokens_out, n_token_capacity, n_token_count_out)`

#### `llama_batch_get_one(tokens, n_tokens)`

Return batch for single sequence of tokens starting at pos\_0

NOTE: this is a helper function to facilitate transition to the new batch API - avoid using it

#### `llama_batch_init(n_tokens, embd, n_seq_max)`

Allocates a batch of tokens on the heap that can hold a maximum of n\_tokens
Each token can be assigned up to n\_seq\_max sequence ids
The batch has to be freed with llama\_batch\_free()
If embd != 0, llama\_batch.embd will be allocated with size of n\_tokens \* embd \* sizeof(float)
Otherwise, llama\_batch.token will be allocated to store n\_tokens llama\_token
The rest of the llama\_batch members are allocated with size n\_tokens
All members are left uninitialized

#### `llama_batch_free(batch)`

Frees a batch of tokens allocated with llama\_batch\_init()

#### `llama_encode(ctx, batch)`

Processes a batch of tokens with the ecoder part of the encoder-decoder model.
Stores the encoder output internally for later use by the decoder cross-attention layers.
0 - success
< 0 - error

#### `llama_decode(ctx, batch)`

Positive return values does not mean a fatal error, but rather a warning.
0 - success
1 - could not find a KV slot for the batch (try reducing the size of the batch or increase the context)
< 0 - error

#### `llama_set_n_threads(ctx, n_threads, n_threads_batch)`

Set the number of threads used for decoding
n\_threads is the number of threads used for generation (single token)
n\_threads\_batch is the number of threads used for prompt and batch processing (multiple tokens)

#### `llama_n_threads(ctx)`

Get the number of threads used for generation of a single token

#### `llama_n_threads_batch(ctx)`

Get the number of threads used for prompt and batch processing (multiple token)

#### `llama_set_embeddings(ctx, embeddings)`

Set whether the model is in embeddings model or not
If true, embeddings will be returned but logits will not

#### `llama_set_causal_attn(ctx, causal_attn)`

Set whether to use causal attention or not
If set to true, the model will only attend to the past tokens

#### `llama_set_warmup(ctx, warmup)`

Set whether the model is in warmup mode or not
If true, all model tensors are activated during llama\_decode() to load and cache their weights.

#### `llama_set_abort_callback(ctx, abort_callback, abort_callback_data)`

Set abort callback

#### `llama_synchronize(ctx)`

Wait until all computations are finished
This is automatically done when using one of the functions below to obtain the computation results
and is not necessary to call it explicitly in most cases

#### `llama_get_logits(ctx)`

Token logits obtained from the last call to llama\_decode()
The logits for which llama\_batch.logits[i] != 0 are stored contiguously
in the order they have appeared in the batch.
Rows: number of tokens for which llama\_batch.logits[i] != 0
Cols: n\_vocab

Returns:

* `CtypesArray[[c_float](https://docs.python.org/3/library/ctypes.html#ctypes.c_float "ctypes.c_float")]`
  –

  Pointer to the logits buffer of shape (n\_tokens, n\_vocab)

#### `llama_get_logits_ith(ctx, i)`

Logits for the ith token. Equivalent to:
llama\_get\_logits(ctx) + i\*n\_vocab

#### `llama_get_embeddings(ctx)`

Get the embeddings for the input
shape: [n\_embd] (1-dimensional)

#### `llama_get_embeddings_ith(ctx, i)`

Get the embeddings for the ith sequence
llama\_get\_embeddings(ctx) + i\*n\_embd

#### `llama_get_embeddings_seq(ctx, seq_id)`

Get the embeddings for a sequence id
Returns NULL if pooling\_type is LLAMA\_POOLING\_TYPE\_NONE
shape: [n\_embd] (1-dimensional)

#### `llama_vocab_get_text(vocab, token)`

#### `llama_vocab_get_score(vocab, token)`

#### `llama_vocab_get_attr(vocab, token)`

#### `llama_vocab_is_eog(vocab, token)`

Check if the token is supposed to end generation (end-of-generation, eg. EOS, EOT, etc.)

#### `llama_vocab_is_control(vocab, token)`

Identify if Token Id is a control token or a render-able token

#### `llama_vocab_bos(vocab)`

beginning-of-sentence

#### `llama_vocab_eos(vocab)`

end-of-sentence

#### `llama_vocab_eot(vocab)`

end-of-turn

#### `llama_vocab_sep(vocab)`

sentence separator

#### `llama_vocab_nl(vocab)`

next-line

#### `llama_vocab_pad(vocab)`

padding

#### `llama_vocab_get_add_bos(vocab)`

#### `llama_vocab_get_add_eos(vocab)`

#### `llama_vocab_fim_pre(vocab)`

#### `llama_vocab_fim_suf(vocab)`

#### `llama_vocab_fim_mid(vocab)`

#### `llama_vocab_fim_pad(vocab)`

#### `llama_vocab_fim_rep(vocab)`

#### `llama_vocab_fim_sep(vocab)`

#### `llama_token_get_text(vocab, token)`

#### `llama_token_get_score(vocab, token)`

#### `llama_token_get_attr(vocab, token)`

#### `llama_token_is_eog(vocab, token)`

#### `llama_token_is_control(vocab, token)`

#### `llama_token_bos(vocab)`

#### `llama_token_eos(vocab)`

#### `llama_token_eot(vocab)`

#### `llama_token_cls(vocab)`

#### `llama_token_sep(vocab)`

#### `llama_token_nl(vocab)`

#### `llama_token_pad(vocab)`

#### `llama_add_bos_token(vocab)`

#### `llama_add_eos_token(vocab)`

#### `llama_token_fim_pre(vocab)`

#### `llama_token_fim_suf(vocab)`

#### `llama_token_fim_mid(vocab)`

#### `llama_token_fim_pad(vocab)`

#### `llama_token_fim_rep(vocab)`

#### `llama_token_fim_sep(vocab)`

#### `llama_vocab_cls(vocab)`

#### `llama_tokenize(vocab, text, text_len, tokens, n_tokens_max, add_special, parse_special)`

Convert the provided text into tokens.

Parameters:

* **`vocab`**
  (`[llama_vocab_p](#llama_cpp.llama_cpp.llama_vocab_p "llama_vocab_p = NewType('llama_vocab_p', int)

        module-attribute
     (llama_cpp.llama_cpp.llama_vocab_p)")`)
  –

  The vocabulary to use for tokenization.
* **`text`**
  (`[bytes](https://docs.python.org/3/library/stdtypes.html#bytes)`)
  –

  The text to tokenize.
* **`text_len`**
  (`[Union](https://docs.python.org/3/library/typing.html#typing.Union "typing.Union")[[c_int](https://docs.python.org/3/library/ctypes.html#ctypes.c_int "ctypes.c_int"), [int](https://docs.python.org/3/library/functions.html#int)]`)
  –

  The length of the text.
* **`tokens`**
  (`CtypesArray[[llama_token](#llama_cpp.llama_cpp.llama_token "llama_token = ctypes.c_int32

        module-attribute
     (llama_cpp.llama_cpp.llama_token)")]`)
  –

  The tokens pointer must be large enough to hold the resulting tokens.
* **`n_max_tokens`**
  –

  The maximum number of tokens to return.
* **`add_special`**
  (`[Union](https://docs.python.org/3/library/typing.html#typing.Union "typing.Union")[[c_bool](https://docs.python.org/3/library/ctypes.html#ctypes.c_bool "ctypes.c_bool"), [bool](https://docs.python.org/3/library/functions.html#bool)]`)
  –

  Allow adding special tokenns if the model is configured to do so.
* **`parse_special`**
  (`[Union](https://docs.python.org/3/library/typing.html#typing.Union "typing.Union")[[c_bool](https://docs.python.org/3/library/ctypes.html#ctypes.c_bool "ctypes.c_bool"), [bool](https://docs.python.org/3/library/functions.html#bool)]`)
  –

  Allow parsing special tokens.

Returns:

* `[int](https://docs.python.org/3/library/functions.html#int)`
  –

  Returns the number of tokens on success, no more than n\_tokens\_max
* `[int](https://docs.python.org/3/library/functions.html#int)`
  –

  Returns a negative number on failure - the number of tokens that would have been returned

#### `llama_token_to_piece(vocab, token, buf, length, lstrip, special)`

Token Id -> Piece.
Uses the vocabulary in the provided context.
Does not write null terminator to the buffer.
User code is responsible to remove the leading whitespace of the first non-BOS token when decoding multiple tokens.

Parameters:

* **`vocab`**
  (`[llama_vocab_p](#llama_cpp.llama_cpp.llama_vocab_p "llama_vocab_p = NewType('llama_vocab_p', int)

        module-attribute
     (llama_cpp.llama_cpp.llama_vocab_p)")`)
  –

  The vocabulary to use for tokenization.
* **`token`**
  (`[Union](https://docs.python.org/3/library/typing.html#typing.Union "typing.Union")[[llama_token](#llama_cpp.llama_cpp.llama_token "llama_token = ctypes.c_int32

        module-attribute
     (llama_cpp.llama_cpp.llama_token)"), [int](https://docs.python.org/3/library/functions.html#int)]`)
  –

  The token to convert.
* **`buf`**
  (`[Union](https://docs.python.org/3/library/typing.html#typing.Union "typing.Union")[[c_char_p](https://docs.python.org/3/library/ctypes.html#ctypes.c_char_p "ctypes.c_char_p"), [bytes](https://docs.python.org/3/library/stdtypes.html#bytes), CtypesArray[[c_char](https://docs.python.org/3/library/ctypes.html#ctypes.c_char "ctypes.c_char")]]`)
  –

  The buffer to write the token to.
* **`length`**
  (`[Union](https://docs.python.org/3/library/typing.html#typing.Union "typing.Union")[[c_int](https://docs.python.org/3/library/ctypes.html#ctypes.c_int "ctypes.c_int"), [int](https://docs.python.org/3/library/functions.html#int)]`)
  –

  The length of the buffer.
* **`lstrip`**
  (`[Union](https://docs.python.org/3/library/typing.html#typing.Union "typing.Union")[[c_int](https://docs.python.org/3/library/ctypes.html#ctypes.c_int "ctypes.c_int"), [int](https://docs.python.org/3/library/functions.html#int)]`)
  –

  The number of leading spaces to skip.
* **`special`**
  (`[Union](https://docs.python.org/3/library/typing.html#typing.Union "typing.Union")[[c_bool](https://docs.python.org/3/library/ctypes.html#ctypes.c_bool "ctypes.c_bool"), [bool](https://docs.python.org/3/library/functions.html#bool)]`)
  –

  If true, special tokens are rendered in the output.

#### `llama_detokenize(model, tokens, n_tokens, text, text_len_max, remove_special, unparse_special)`

Convert the provided tokens into text (inverse of llama\_tokenize()).

Parameters:

* **`model`**
  (`[llama_model_p](#llama_cpp.llama_cpp.llama_model_p "llama_model_p = NewType('llama_model_p', int)

        module-attribute
     (llama_cpp.llama_cpp.llama_model_p)")`)
  –

  The model to use for tokenization.
* **`tokens`**
  (`CtypesArray[[llama_token](#llama_cpp.llama_cpp.llama_token "llama_token = ctypes.c_int32

        module-attribute
     (llama_cpp.llama_cpp.llama_token)")]`)
  –

  The tokens to convert.
* **`n_tokens`**
  (`[Union](https://docs.python.org/3/library/typing.html#typing.Union "typing.Union")[[c_int](https://docs.python.org/3/library/ctypes.html#ctypes.c_int "ctypes.c_int"), [int](https://docs.python.org/3/library/functions.html#int)]`)
  –

  The number of tokens.
* **`text`**
  (`[bytes](https://docs.python.org/3/library/stdtypes.html#bytes)`)
  –

  The buffer to write the text to.
* **`text_len_max`**
  (`[Union](https://docs.python.org/3/library/typing.html#typing.Union "typing.Union")[[c_int](https://docs.python.org/3/library/ctypes.html#ctypes.c_int "ctypes.c_int"), [int](https://docs.python.org/3/library/functions.html#int)]`)
  –

  The length of the buffer.
* **`remove_special`**
  (`[Union](https://docs.python.org/3/library/typing.html#typing.Union "typing.Union")[[c_bool](https://docs.python.org/3/library/ctypes.html#ctypes.c_bool "ctypes.c_bool"), [bool](https://docs.python.org/3/library/functions.html#bool)]`)
  –

  Allow to remove BOS and EOS tokens if model is configured to do so.
* **`unparse_special`**
  (`[Union](https://docs.python.org/3/library/typing.html#typing.Union "typing.Union")[[c_bool](https://docs.python.org/3/library/ctypes.html#ctypes.c_bool "ctypes.c_bool"), [bool](https://docs.python.org/3/library/functions.html#bool)]`)
  –

  If true, special tokens are rendered in the output.

#### `llama_chat_apply_template(tmpl, chat, n_msg, add_ass, buf, length)`

Apply chat template.

Parameters:

* **`tmpl`**
  (`[bytes](https://docs.python.org/3/library/stdtypes.html#bytes)`)
  –

  Template to use. If None, uses model's default
* **`chat`**
  (`CtypesArray[[llama_chat_message](#llama_cpp.llama_cpp.llama_chat_message "llama_chat_message (llama_cpp.llama_cpp.llama_chat_message)")]`)
  –

  Array of chat messages
* **`n_msg`**
  (`[int](https://docs.python.org/3/library/functions.html#int)`)
  –

  Number of messages
* **`add_ass`**
  (`[bool](https://docs.python.org/3/library/functions.html#bool)`)
  –

  Whether to end prompt with assistant token
* **`buf`**
  (`[bytes](https://docs.python.org/3/library/stdtypes.html#bytes)`)
  –

  Output buffer
* **`length`**
  (`[int](https://docs.python.org/3/library/functions.html#int)`)
  –

  Buffer length

Returns:

* `[int](https://docs.python.org/3/library/functions.html#int)`
  –

  Number of bytes written, or needed if buffer too small

#### `llama_chat_builtin_templates(output, len)`

Get list of built-in chat templates.

Parameters:

* **`output`**
  (`CtypesArray[[bytes](https://docs.python.org/3/library/stdtypes.html#bytes)]`)
  –

  Output buffer to store template names.
* **`len`**
  (`[Union](https://docs.python.org/3/library/typing.html#typing.Union "typing.Union")[[c_size_t](https://docs.python.org/3/library/ctypes.html#ctypes.c_size_t "ctypes.c_size_t"), [int](https://docs.python.org/3/library/functions.html#int)]`)
  –

  Length of the output buffer.

Returns:

* `[int](https://docs.python.org/3/library/functions.html#int)`
  –

  Number of templates available.
* `[int](https://docs.python.org/3/library/functions.html#int)`
  –

  Returns a negative number on error.

#### `llama_sampler_context_t = ctypes.c_void_p` `module-attribute`

#### `llama_sampler_i`

Bases: `[Structure](https://docs.python.org/3/library/ctypes.html#ctypes.Structure "ctypes.Structure")`

#### `llama_sampler`

Bases: `[Structure](https://docs.python.org/3/library/ctypes.html#ctypes.Structure "ctypes.Structure")`

#### `llama_sampler_p = CtypesPointer[llama_sampler]` `module-attribute`

#### `llama_sampler_p_ctypes = ctypes.POINTER(llama_sampler)` `module-attribute`

#### `llama_sampler_i_name = ctypes.CFUNCTYPE(ctypes.c_char_p, llama_sampler_p_ctypes)` `module-attribute`

#### `llama_sampler_i_accept = ctypes.CFUNCTYPE(None, llama_sampler_p_ctypes, llama_token)` `module-attribute`

#### `llama_sampler_i_apply = ctypes.CFUNCTYPE(None, llama_sampler_p_ctypes, llama_token_data_array_p)` `module-attribute`

#### `llama_sampler_i_reset = ctypes.CFUNCTYPE(None, llama_sampler_p_ctypes)` `module-attribute`

#### `llama_sampler_i_clone = ctypes.CFUNCTYPE(llama_sampler_p_ctypes, llama_sampler_p_ctypes)` `module-attribute`

#### `llama_sampler_i_free = ctypes.CFUNCTYPE(None, llama_sampler_p_ctypes)` `module-attribute`

#### `llama_sampler_init(iface, ctx)`

#### `llama_sampler_name(smpl)`

#### `llama_sampler_accept(smpl, token)`

#### `llama_sampler_apply(smpl, cur_p)`

#### `llama_sampler_reset(smpl)`

#### `llama_sampler_clone(smpl)`

#### `llama_sampler_free(smpl)`

#### `llama_sampler_chain_init(params)`

#### `llama_sampler_chain_add(chain, smpl)`

#### `llama_sampler_chain_get(chain, i)`

#### `llama_sampler_chain_n(chain)`

#### `llama_sampler_chain_remove(chain, i)`

#### `llama_sampler_init_greedy()`

#### `llama_sampler_init_dist(seed)`

#### `llama_sampler_init_softmax()`

#### `llama_sampler_init_top_k(k)`

#### `llama_sampler_init_top_p(p, min_keep)`

#### `llama_sampler_init_min_p(p, min_keep)`

#### `llama_sampler_init_typical(p, min_keep)`

#### `llama_sampler_init_temp(t)`

#### `llama_sampler_init_temp_ext(t, delta, exponent)`

#### `llama_sampler_init_xtc(p, t, min_keep, seed)`

#### `llama_sampler_init_top_n_sigma(n)`

#### `llama_sampler_init_mirostat(n_vocab, seed, tau, eta, m)`

#### `llama_sampler_init_mirostat_v2(seed, tau, eta)`

#### `llama_sampler_init_grammar(vocab, grammar_str, grammar_root)`

#### `llama_sampler_init_grammar_lazy_patterns(vocab, grammar_str, grammar_root, trigger_patterns, num_trigger_patterns, trigger_tokens, num_trigger_tokens)`

#### `llama_sampler_init_penalties(penalty_last_n, penalty_repeat, penalty_freq, penalty_present)`

#### `llama_sampler_init_dry(vocab, n_ctx_train, dry_multiplier, dry_base, dry_allowed_length, dry_penalty_last_n, seq_breakers, num_breakers)`

#### `llama_sampler_init_logit_bias(n_vocab, n_logit_bias, logit_bias)`

#### `llama_sampler_init_infill(vocab)`

#### `llama_sampler_get_seed(smpl)`

#### `llama_sampler_sample(smpl, ctx, idx)`

#### `llama_split_path(split_path, maxlen, path_prefix, split_no, split_count)`

Build a split GGUF final path for this chunk.

#### `llama_split_prefix(split_prefix, maxlen, split_path, split_no, split_count)`

Extract the path prefix from the split\_path if and only if the split\_no and split\_count match.

#### `llama_print_system_info()`

#### `llama_log_set(log_callback, user_data)`

Set callback for all future logging events.

If this is not called, or NULL is supplied, everything is output on stderr.

#### `llama_perf_context_data`

Bases: `[Structure](https://docs.python.org/3/library/ctypes.html#ctypes.Structure "ctypes.Structure")`

#### `llama_perf_sampler_data`

Bases: `[Structure](https://docs.python.org/3/library/ctypes.html#ctypes.Structure "ctypes.Structure")`

#### `llama_perf_context(ctx)`

#### `llama_perf_context_print(ctx)`

#### `llama_perf_context_reset(ctx)`

#### `llama_perf_sampler(chain)`

#### `llama_perf_sampler_print(chain)`

#### `llama_perf_sampler_reset(chain)`

#### `LLAMA_MAX_DEVICES = _lib.llama_max_devices()` `module-attribute`

#### `LLAMA_DEFAULT_SEED = 4294967295` `module-attribute`

#### `LLAMA_TOKEN_NULL = -1` `module-attribute`

#### `LLAMA_FILE_MAGIC_GGLA = 1734831201` `module-attribute`

#### `LLAMA_FILE_MAGIC_GGSN = 1734833006` `module-attribute`

#### `LLAMA_FILE_MAGIC_GGSQ = 1734833009` `module-attribute`

#### `LLAMA_SESSION_MAGIC = LLAMA_FILE_MAGIC_GGSN` `module-attribute`

#### `LLAMA_SESSION_VERSION = 9` `module-attribute`

#### `LLAMA_STATE_SEQ_MAGIC = LLAMA_FILE_MAGIC_GGSQ` `module-attribute`

#### `LLAMA_STATE_SEQ_VERSION = 2` `module-attribute`

#### `LLAMA_VOCAB_TYPE_NONE = 0` `module-attribute`

For models without vocab

#### `LLAMA_VOCAB_TYPE_SPM = 1` `module-attribute`

LLaMA tokenizer based on byte-level BPE with byte fallback

#### `LLAMA_VOCAB_TYPE_BPE = 2` `module-attribute`

GPT-2 tokenizer based on byte-level BPE

#### `LLAMA_VOCAB_TYPE_WPM = 3` `module-attribute`

BERT tokenizer based on WordPiece

#### `LLAMA_VOCAB_TYPE_UGM = 4` `module-attribute`

T5 tokenizer based on Unigram

#### `LLAMA_VOCAB_TYPE_RWKV = 5` `module-attribute`

RWKV tokenizer based on greedy tokenization

#### `LLAMA_VOCAB_PRE_TYPE_DEFAULT = 0` `module-attribute`

#### `LLAMA_VOCAB_PRE_TYPE_LLAMA3 = 1` `module-attribute`

#### `LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_LLM = 2` `module-attribute`

#### `LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_CODER = 3` `module-attribute`

#### `LLAMA_VOCAB_PRE_TYPE_FALCON = 4` `module-attribute`

#### `LLAMA_VOCAB_PRE_TYPE_MPT = 5` `module-attribute`

#### `LLAMA_VOCAB_PRE_TYPE_STARCODER = 6` `module-attribute`

#### `LLAMA_VOCAB_PRE_TYPE_GPT2 = 7` `module-attribute`

#### `LLAMA_VOCAB_PRE_TYPE_REFACT = 8` `module-attribute`

#### `LLAMA_VOCAB_PRE_TYPE_COMMAND_R = 9` `module-attribute`

#### `LLAMA_VOCAB_PRE_TYPE_STABLELM2 = 10` `module-attribute`

#### `LLAMA_VOCAB_PRE_TYPE_QWEN2 = 11` `module-attribute`

#### `LLAMA_VOCAB_PRE_TYPE_OLMO = 12` `module-attribute`

#### `LLAMA_VOCAB_PRE_TYPE_DBRX = 13` `module-attribute`

#### `LLAMA_VOCAB_PRE_TYPE_SMAUG = 14` `module-attribute`

#### `LLAMA_VOCAB_PRE_TYPE_PORO = 15` `module-attribute`

#### `LLAMA_VOCAB_PRE_TYPE_CHATGLM3 = 16` `module-attribute`

#### `LLAMA_VOCAB_PRE_TYPE_CHATGLM4 = 17` `module-attribute`

#### `LLAMA_VOCAB_PRE_TYPE_VIKING = 18` `module-attribute`

#### `LLAMA_VOCAB_PRE_TYPE_JAIS = 19` `module-attribute`

#### `LLAMA_VOCAB_PRE_TYPE_TEKKEN = 20` `module-attribute`

#### `LLAMA_VOCAB_PRE_TYPE_SMOLLM = 21` `module-attribute`

#### `LLAMA_VOCAB_PRE_TYPE_CODESHELL = 22` `module-attribute`

#### `LLAMA_VOCAB_PRE_TYPE_BLOOM = 23` `module-attribute`

#### `LLAMA_VOCAB_PRE_TYPE_GPT3_FINNISH = 24` `module-attribute`

#### `LLAMA_VOCAB_PRE_TYPE_EXAONE = 25` `module-attribute`

#### `LLAMA_VOCAB_PRE_TYPE_CHAMELEON = 26` `module-attribute`

#### `LLAMA_VOCAB_PRE_TYPE_MINERVA = 27` `module-attribute`

#### `LLAMA_VOCAB_PRE_TYPE_DEEPSEEK3_LLM = 28` `module-attribute`

#### `LLAMA_VOCAB_PRE_TYPE_GPT4O = 29` `module-attribute`

#### `LLAMA_VOCAB_PRE_TYPE_SUPERBPE = 30` `module-attribute`

#### `LLAMA_VOCAB_PRE_TYPE_TRILLION = 31` `module-attribute`

#### `LLAMA_VOCAB_PRE_TYPE_BAILINGMOE = 32` `module-attribute`

#### `LLAMA_VOCAB_PRE_TYPE_LLAMA4 = 33` `module-attribute`

#### `LLAMA_VOCAB_PRE_TYPE_PIXTRAL = 34` `module-attribute`

#### `LLAMA_ROPE_TYPE_NONE = -1` `module-attribute`

#### `LLAMA_ROPE_TYPE_NORM = 0` `module-attribute`

#### `LLAMA_ROPE_TYPE_NEOX = 2` `module-attribute`

#### `LLAMA_ROPE_TYPE_MROPE = 8` `module-attribute`

#### `LLAMA_ROPE_TYPE_VISION = 24` `module-attribute`

#### `LLAMA_TOKEN_TYPE_UNDEFINED = 0` `module-attribute`

#### `LLAMA_TOKEN_TYPE_NORMAL = 1` `module-attribute`

#### `LLAMA_TOKEN_TYPE_UNKNOWN = 2` `module-attribute`

#### `LLAMA_TOKEN_TYPE_CONTROL = 3` `module-attribute`

#### `LLAMA_TOKEN_TYPE_USER_DEFINED = 4` `module-attribute`

#### `LLAMA_TOKEN_TYPE_UNUSED = 5` `module-attribute`

#### `LLAMA_TOKEN_TYPE_BYTE = 6` `module-attribute`

#### `LLAMA_TOKEN_ATTR_UNDEFINED = 0` `module-attribute`

#### `LLAMA_TOKEN_ATTR_UNKNOWN = 1 << 0` `module-attribute`

#### `LLAMA_TOKEN_ATTR_UNUSED = 1 << 1` `module-attribute`

#### `LLAMA_TOKEN_ATTR_NORMAL = 1 << 2` `module-attribute`

#### `LLAMA_TOKEN_ATTR_CONTROL = 1 << 3` `module-attribute`

#### `LLAMA_TOKEN_ATTR_USER_DEFINED = 1 << 4` `module-attribute`

#### `LLAMA_TOKEN_ATTR_BYTE = 1 << 5` `module-attribute`

#### `LLAMA_TOKEN_ATTR_NORMALIZED = 1 << 6` `module-attribute`

#### `LLAMA_TOKEN_ATTR_LSTRIP = 1 << 7` `module-attribute`

#### `LLAMA_TOKEN_ATTR_RSTRIP = 1 << 8` `module-attribute`

#### `LLAMA_TOKEN_ATTR_SINGLE_WORD = 1 << 9` `module-attribute`

#### `LLAMA_FTYPE_ALL_F32 = 0` `module-attribute`

#### `LLAMA_FTYPE_MOSTLY_F16 = 1` `module-attribute`

#### `LLAMA_FTYPE_MOSTLY_Q4_0 = 2` `module-attribute`

#### `LLAMA_FTYPE_MOSTLY_Q4_1 = 3` `module-attribute`

#### `LLAMA_FTYPE_MOSTLY_Q8_0 = 7` `module-attribute`

#### `LLAMA_FTYPE_MOSTLY_Q5_0 = 8` `module-attribute`

#### `LLAMA_FTYPE_MOSTLY_Q5_1 = 9` `module-attribute`

#### `LLAMA_FTYPE_MOSTLY_Q2_K = 10` `module-attribute`

#### `LLAMA_FTYPE_MOSTLY_Q3_K_S = 11` `module-attribute`

#### `LLAMA_FTYPE_MOSTLY_Q3_K_M = 12` `module-attribute`

#### `LLAMA_FTYPE_MOSTLY_Q3_K_L = 13` `module-attribute`

#### `LLAMA_FTYPE_MOSTLY_Q4_K_S = 14` `module-attribute`

#### `LLAMA_FTYPE_MOSTLY_Q4_K_M = 15` `module-attribute`

#### `LLAMA_FTYPE_MOSTLY_Q5_K_S = 16` `module-attribute`

#### `LLAMA_FTYPE_MOSTLY_Q5_K_M = 17` `module-attribute`

#### `LLAMA_FTYPE_MOSTLY_Q6_K = 18` `module-attribute`

#### `LLAMA_FTYPE_MOSTLY_IQ2_XXS = 19` `module-attribute`

#### `LLAMA_FTYPE_MOSTLY_IQ2_XS = 20` `module-attribute`

#### `LLAMA_FTYPE_MOSTLY_Q2_K_S = 21` `module-attribute`

#### `LLAMA_FTYPE_MOSTLY_IQ3_XS = 22` `module-attribute`

#### `LLAMA_FTYPE_MOSTLY_IQ3_XXS = 23` `module-attribute`

#### `LLAMA_FTYPE_MOSTLY_IQ1_S = 24` `module-attribute`

#### `LLAMA_FTYPE_MOSTLY_IQ4_NL = 25` `module-attribute`

#### `LLAMA_FTYPE_MOSTLY_IQ3_S = 26` `module-attribute`

#### `LLAMA_FTYPE_MOSTLY_IQ3_M = 27` `module-attribute`

#### `LLAMA_FTYPE_MOSTLY_IQ2_S = 28` `module-attribute`

#### `LLAMA_FTYPE_MOSTLY_IQ2_M = 29` `module-attribute`

#### `LLAMA_FTYPE_MOSTLY_IQ4_XS = 30` `module-attribute`

#### `LLAMA_FTYPE_MOSTLY_IQ1_M = 31` `module-attribute`

#### `LLAMA_FTYPE_MOSTLY_BF16 = 32` `module-attribute`

#### `LLAMA_FTYPE_MOSTLY_TQ1_0 = 36` `module-attribute`

#### `LLAMA_FTYPE_MOSTLY_TQ2_0 = 37` `module-attribute`

#### `LLAMA_FTYPE_GUESSED = 1024` `module-attribute`

#### `LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED = -1` `module-attribute`

#### `LLAMA_ROPE_SCALING_TYPE_NONE = 0` `module-attribute`

#### `LLAMA_ROPE_SCALING_TYPE_LINEAR = 1` `module-attribute`

#### `LLAMA_ROPE_SCALING_TYPE_YARN = 2` `module-attribute`

#### `LLAMA_ROPE_SCALING_TYPE_LONGROPE = 3` `module-attribute`

#### `LLAMA_ROPE_SCALING_TYPE_MAX_VALUE = LLAMA_ROPE_SCALING_TYPE_YARN` `module-attribute`

#### `LLAMA_POOLING_TYPE_UNSPECIFIED = -1` `module-attribute`

#### `LLAMA_POOLING_TYPE_NONE = 0` `module-attribute`

#### `LLAMA_POOLING_TYPE_MEAN = 1` `module-attribute`

#### `LLAMA_POOLING_TYPE_CLS = 2` `module-attribute`

#### `LLAMA_POOLING_TYPE_LAST = 3` `module-attribute`

#### `LLAMA_POOLING_TYPE_RANK = 4` `module-attribute`

#### `LLAMA_ATTENTION_TYPE_UNSPECIFIED = -1` `module-attribute`

#### `LLAMA_ATTENTION_TYPE_CAUSAL = 0` `module-attribute`

#### `LLAMA_ATTENTION_TYPE_NON_CAUSAL = 1` `module-attribute`

#### `LLAMA_SPLIT_MODE_NONE = 0` `module-attribute`

#### `LLAMA_SPLIT_MODE_LAYER = 1` `module-attribute`

#### `LLAMA_SPLIT_MODE_ROW = 2` `module-attribute`

#### `LLAMA_KV_OVERRIDE_TYPE_INT = 0` `module-attribute`

#### `LLAMA_KV_OVERRIDE_TYPE_FLOAT = 1` `module-attribute`

#### `LLAMA_KV_OVERRIDE_TYPE_BOOL = 2` `module-attribute`

#### `LLAMA_KV_OVERRIDE_TYPE_STR = 3` `module-attribute`

## Misc

### `llama_cpp.llama_types`

Types and request signatures for OpenAI compatibility

NOTE: These types may change to match the OpenAI OpenAPI specification.

Based on the OpenAI OpenAPI specification:
<https://github.com/openai/openai-openapi/blob/master/openapi.yaml>

#### `JsonType = Union[None, int, str, bool, List[Any], Dict[str, Any]]` `module-attribute`

#### `EmbeddingUsage`

Bases: `TypedDict`

##### `prompt_tokens` `instance-attribute`

##### `total_tokens` `instance-attribute`

#### `Embedding`

Bases: `TypedDict`

##### `index` `instance-attribute`

##### `object` `instance-attribute`

##### `embedding` `instance-attribute`

#### `CreateEmbeddingResponse`

Bases: `TypedDict`

##### `object` `instance-attribute`

##### `model` `instance-attribute`

##### `data` `instance-attribute`

##### `usage` `instance-attribute`

#### `CompletionLogprobs`

Bases: `TypedDict`

##### `text_offset` `instance-attribute`

##### `token_logprobs` `instance-attribute`

##### `tokens` `instance-attribute`

##### `top_logprobs` `instance-attribute`

#### `CompletionChoice`

Bases: `TypedDict`

##### `text` `instance-attribute`

##### `index` `instance-attribute`

##### `logprobs` `instance-attribute`

##### `finish_reason` `instance-attribute`

#### `CompletionUsage`

Bases: `TypedDict`

##### `prompt_tokens` `instance-attribute`

##### `completion_tokens` `instance-attribute`

##### `total_tokens` `instance-attribute`

#### `CreateCompletionResponse`

Bases: `TypedDict`

##### `id` `instance-attribute`

##### `object` `instance-attribute`

##### `created` `instance-attribute`

##### `model` `instance-attribute`

##### `choices` `instance-attribute`

##### `usage` `instance-attribute`

#### `ChatCompletionResponseFunctionCall`

Bases: `TypedDict`

##### `name` `instance-attribute`

##### `arguments` `instance-attribute`

#### `ChatCompletionResponseMessage`

Bases: `TypedDict`

##### `content` `instance-attribute`

##### `tool_calls` `instance-attribute`

##### `role` `instance-attribute`

##### `function_call` `instance-attribute`

#### `ChatCompletionFunction`

Bases: `TypedDict`

##### `name` `instance-attribute`

##### `description` `instance-attribute`

##### `parameters` `instance-attribute`

#### `ChatCompletionTopLogprobToken`

Bases: `TypedDict`

##### `token` `instance-attribute`

##### `logprob` `instance-attribute`

##### `bytes` `instance-attribute`

#### `ChatCompletionLogprobToken`

Bases: `[ChatCompletionTopLogprobToken](#llama_cpp.llama_types.ChatCompletionTopLogprobToken "ChatCompletionTopLogprobToken (llama_cpp.llama_types.ChatCompletionTopLogprobToken)")`

##### `token` `instance-attribute`

##### `logprob` `instance-attribute`

##### `bytes` `instance-attribute`

##### `top_logprobs` `instance-attribute`

#### `ChatCompletionLogprobs`

Bases: `TypedDict`| ```      ``` | ``` class ChatCompletionLogprobs(TypedDict):     content: Optional[List[ChatCompletionLogprobToken]]     refusal: Optional[List[ChatCompletionLogprobToken]]  ``` |
| --- | --- |

##### `content` `instance-attribute`

##### `refusal` `instance-attribute`

#### `ChatCompletionResponseChoice`

Bases: `TypedDict`

##### `index` `instance-attribute`

##### `message` `instance-attribute`

##### `logprobs` `instance-attribute`

##### `finish_reason` `instance-attribute`

#### `CreateChatCompletionResponse`

Bases: `TypedDict`

##### `id` `instance-attribute`

##### `object` `instance-attribute`

##### `created` `instance-attribute`

##### `model` `instance-attribute`

##### `choices` `instance-attribute`

##### `usage` `instance-attribute`

#### `ChatCompletionMessageToolCallChunkFunction`

Bases: `TypedDict`

##### `name` `instance-attribute`

##### `arguments` `instance-attribute`

#### `ChatCompletionMessageToolCallChunk`

Bases: `TypedDict`

##### `index` `instance-attribute`

##### `id` `instance-attribute`

##### `type` `instance-attribute`

##### `function` `instance-attribute`

#### `ChatCompletionStreamResponseDeltaEmpty`

Bases: `TypedDict`

#### `ChatCompletionStreamResponseDeltaFunctionCall`

Bases: `TypedDict`

##### `name` `instance-attribute`

##### `arguments` `instance-attribute`

#### `ChatCompletionStreamResponseDelta`

Bases: `TypedDict`

##### `content` `instance-attribute`

##### `function_call` `instance-attribute`

##### `tool_calls` `instance-attribute`

##### `role` `instance-attribute`

#### `ChatCompletionStreamResponseChoice`

Bases: `TypedDict`

##### `index` `instance-attribute`

##### `delta` `instance-attribute`

##### `finish_reason` `instance-attribute`

##### `logprobs` `instance-attribute`

#### `CreateChatCompletionStreamResponse`

Bases: `TypedDict`

##### `id` `instance-attribute`

##### `model` `instance-attribute`

##### `object` `instance-attribute`

##### `created` `instance-attribute`

##### `choices` `instance-attribute`

#### `ChatCompletionFunctions`

Bases: `TypedDict`

##### `name` `instance-attribute`

##### `description` `instance-attribute`

##### `parameters` `instance-attribute`

#### `ChatCompletionFunctionCallOption`

Bases: `TypedDict`

##### `name` `instance-attribute`

#### `ChatCompletionRequestResponseFormat`

Bases: `TypedDict`

##### `type` `instance-attribute`

##### `schema` `instance-attribute`

#### `ChatCompletionRequestMessageContentPartText`

Bases: `TypedDict`

##### `type` `instance-attribute`

##### `text` `instance-attribute`

#### `ChatCompletionRequestMessageContentPartImageImageUrl`

Bases: `TypedDict`

##### `url` `instance-attribute`

##### `detail` `instance-attribute`

#### `ChatCompletionRequestMessageContentPartImage`

Bases: `TypedDict`

##### `type` `instance-attribute`

##### `image_url` `instance-attribute`

#### `ChatCompletionRequestMessageContentPart = Union[ChatCompletionRequestMessageContentPartText, ChatCompletionRequestMessageContentPartImage]` `module-attribute`

#### `ChatCompletionRequestSystemMessage`

Bases: `TypedDict`

##### `role` `instance-attribute`

##### `content` `instance-attribute`

#### `ChatCompletionRequestUserMessage`

Bases: `TypedDict`

##### `role` `instance-attribute`

##### `content` `instance-attribute`

#### `ChatCompletionMessageToolCallFunction`

Bases: `TypedDict`

##### `name` `instance-attribute`

##### `arguments` `instance-attribute`

#### `ChatCompletionMessageToolCall`

Bases: `TypedDict`

##### `id` `instance-attribute`

##### `type` `instance-attribute`

##### `function` `instance-attribute`

#### `ChatCompletionMessageToolCalls = List[ChatCompletionMessageToolCall]` `module-attribute`

#### `ChatCompletionRequestAssistantMessageFunctionCall`

Bases: `TypedDict`

##### `name` `instance-attribute`

##### `arguments` `instance-attribute`

#### `ChatCompletionRequestAssistantMessage`

Bases: `TypedDict`

##### `role` `instance-attribute`

##### `content` `instance-attribute`

##### `tool_calls` `instance-attribute`

##### `function_call` `instance-attribute`

#### `ChatCompletionRequestToolMessage`

Bases: `TypedDict`

##### `role` `instance-attribute`

##### `content` `instance-attribute`

##### `tool_call_id` `instance-attribute`

#### `ChatCompletionRequestFunctionMessage`

Bases: `TypedDict`

##### `role` `instance-attribute`

##### `content` `instance-attribute`

##### `name` `instance-attribute`

#### `ChatCompletionRequestMessage = Union[ChatCompletionRequestSystemMessage, ChatCompletionRequestUserMessage, ChatCompletionRequestAssistantMessage, ChatCompletionRequestUserMessage, ChatCompletionRequestToolMessage, ChatCompletionRequestFunctionMessage]` `module-attribute`

#### `ChatCompletionRequestFunctionCallOption`

Bases: `TypedDict`

##### `name` `instance-attribute`

#### `ChatCompletionRequestFunctionCall = Union[Literal['none', 'auto'], ChatCompletionRequestFunctionCallOption]` `module-attribute`

#### `ChatCompletionFunctionParameters = Dict[str, JsonType]` `module-attribute`

#### `ChatCompletionToolFunction`

Bases: `TypedDict`

##### `name` `instance-attribute`

##### `description` `instance-attribute`

##### `parameters` `instance-attribute`

#### `ChatCompletionTool`

Bases: `TypedDict`

##### `type` `instance-attribute`

##### `function` `instance-attribute`

#### `ChatCompletionNamedToolChoiceFunction`

Bases: `TypedDict`

##### `name` `instance-attribute`

#### `ChatCompletionNamedToolChoice`

Bases: `TypedDict`

##### `type` `instance-attribute`

##### `function` `instance-attribute`

#### `ChatCompletionToolChoiceOption = Union[Literal['none', 'auto', 'required'], ChatCompletionNamedToolChoice]` `module-attribute`

#### `EmbeddingData = Embedding` `module-attribute`

#### `CompletionChunk = CreateCompletionResponse` `module-attribute`

#### `Completion = CreateCompletionResponse` `module-attribute`

#### `CreateCompletionStreamResponse = CreateCompletionResponse` `module-attribute`

#### `ChatCompletionMessage = ChatCompletionResponseMessage` `module-attribute`

#### `ChatCompletionChoice = ChatCompletionResponseChoice` `module-attribute`

#### `ChatCompletion = CreateChatCompletionResponse` `module-attribute`

#### `ChatCompletionChunkDeltaEmpty = ChatCompletionStreamResponseDeltaEmpty` `module-attribute`

#### `ChatCompletionChunkChoice = ChatCompletionStreamResponseChoice` `module-attribute`

#### `ChatCompletionChunkDelta = ChatCompletionStreamResponseDelta` `module-attribute`

#### `ChatCompletionChunk = CreateChatCompletionStreamResponse` `module-attribute`

#### `ChatCompletionStreamResponse = CreateChatCompletionStreamResponse` `module-attribute`

#### `ChatCompletionResponseFunction = ChatCompletionFunction` `module-attribute`

#### `ChatCompletionFunctionCall = ChatCompletionResponseFunctionCall` `module-attribute`
```

### llama.h

```cpp
#ifndef LLAMA_H
#define LLAMA_H

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-opt.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>

#ifdef LLAMA_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef LLAMA_BUILD
#            define LLAMA_API __declspec(dllexport)
#        else
#            define LLAMA_API __declspec(dllimport)
#        endif
#    else
#        define LLAMA_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define LLAMA_API
#endif

#ifdef __GNUC__
#    define DEPRECATED(func, hint) func __attribute__((deprecated(hint)))
#elif defined(_MSC_VER)
#    define DEPRECATED(func, hint) __declspec(deprecated(hint)) func
#else
#    define DEPRECATED(func, hint) func
#endif

#define LLAMA_DEFAULT_SEED 0xFFFFFFFF

#define LLAMA_TOKEN_NULL -1

#define LLAMA_FILE_MAGIC_GGLA 0x67676c61u // 'ggla'
#define LLAMA_FILE_MAGIC_GGSN 0x6767736eu // 'ggsn'
#define LLAMA_FILE_MAGIC_GGSQ 0x67677371u // 'ggsq'

#define LLAMA_SESSION_MAGIC   LLAMA_FILE_MAGIC_GGSN
#define LLAMA_SESSION_VERSION 9

#define LLAMA_STATE_SEQ_MAGIC   LLAMA_FILE_MAGIC_GGSQ
#define LLAMA_STATE_SEQ_VERSION 2

#ifdef __cplusplus
extern "C" {
#endif

    //
    // C interface
    //
    // TODO: show sample usage
    //

    struct llama_vocab;
    struct llama_model;
    struct llama_context;
    struct llama_sampler;
    struct llama_kv_cache;

    typedef int32_t llama_pos;
    typedef int32_t llama_token;
    typedef int32_t llama_seq_id;

    enum llama_vocab_type {
        LLAMA_VOCAB_TYPE_NONE = 0, // For models without vocab
        LLAMA_VOCAB_TYPE_SPM  = 1, // LLaMA tokenizer based on byte-level BPE with byte fallback
        LLAMA_VOCAB_TYPE_BPE  = 2, // GPT-2 tokenizer based on byte-level BPE
        LLAMA_VOCAB_TYPE_WPM  = 3, // BERT tokenizer based on WordPiece
        LLAMA_VOCAB_TYPE_UGM  = 4, // T5 tokenizer based on Unigram
        LLAMA_VOCAB_TYPE_RWKV = 5, // RWKV tokenizer based on greedy tokenization
    };

    // pre-tokenization types
    enum llama_vocab_pre_type {
        LLAMA_VOCAB_PRE_TYPE_DEFAULT        = 0,
        LLAMA_VOCAB_PRE_TYPE_LLAMA3         = 1,
        LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_LLM   = 2,
        LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_CODER = 3,
        LLAMA_VOCAB_PRE_TYPE_FALCON         = 4,
        LLAMA_VOCAB_PRE_TYPE_MPT            = 5,
        LLAMA_VOCAB_PRE_TYPE_STARCODER      = 6,
        LLAMA_VOCAB_PRE_TYPE_GPT2           = 7,
        LLAMA_VOCAB_PRE_TYPE_REFACT         = 8,
        LLAMA_VOCAB_PRE_TYPE_COMMAND_R      = 9,
        LLAMA_VOCAB_PRE_TYPE_STABLELM2      = 10,
        LLAMA_VOCAB_PRE_TYPE_QWEN2          = 11,
        LLAMA_VOCAB_PRE_TYPE_OLMO           = 12,
        LLAMA_VOCAB_PRE_TYPE_DBRX           = 13,
        LLAMA_VOCAB_PRE_TYPE_SMAUG          = 14,
        LLAMA_VOCAB_PRE_TYPE_PORO           = 15,
        LLAMA_VOCAB_PRE_TYPE_CHATGLM3       = 16,
        LLAMA_VOCAB_PRE_TYPE_CHATGLM4       = 17,
        LLAMA_VOCAB_PRE_TYPE_VIKING         = 18,
        LLAMA_VOCAB_PRE_TYPE_JAIS           = 19,
        LLAMA_VOCAB_PRE_TYPE_TEKKEN         = 20,
        LLAMA_VOCAB_PRE_TYPE_SMOLLM         = 21,
        LLAMA_VOCAB_PRE_TYPE_CODESHELL      = 22,
        LLAMA_VOCAB_PRE_TYPE_BLOOM          = 23,
        LLAMA_VOCAB_PRE_TYPE_GPT3_FINNISH   = 24,
        LLAMA_VOCAB_PRE_TYPE_EXAONE         = 25,
        LLAMA_VOCAB_PRE_TYPE_CHAMELEON      = 26,
        LLAMA_VOCAB_PRE_TYPE_MINERVA        = 27,
        LLAMA_VOCAB_PRE_TYPE_DEEPSEEK3_LLM  = 28,
        LLAMA_VOCAB_PRE_TYPE_GPT4O          = 29,
        LLAMA_VOCAB_PRE_TYPE_SUPERBPE       = 30,
        LLAMA_VOCAB_PRE_TYPE_TRILLION       = 31,
        LLAMA_VOCAB_PRE_TYPE_BAILINGMOE     = 32,
        LLAMA_VOCAB_PRE_TYPE_LLAMA4         = 33,
        LLAMA_VOCAB_PRE_TYPE_PIXTRAL        = 34,
        LLAMA_VOCAB_PRE_TYPE_SEED_CODER     = 35,
    };

    enum llama_rope_type {
        LLAMA_ROPE_TYPE_NONE   = -1,
        LLAMA_ROPE_TYPE_NORM   = 0,
        LLAMA_ROPE_TYPE_NEOX   = GGML_ROPE_TYPE_NEOX,
        LLAMA_ROPE_TYPE_MROPE  = GGML_ROPE_TYPE_MROPE,
        LLAMA_ROPE_TYPE_VISION = GGML_ROPE_TYPE_VISION,
    };

    enum llama_token_type { //TODO: remove, required until per token attributes are available from GGUF file
        LLAMA_TOKEN_TYPE_UNDEFINED    = 0,
        LLAMA_TOKEN_TYPE_NORMAL       = 1,
        LLAMA_TOKEN_TYPE_UNKNOWN      = 2,
        LLAMA_TOKEN_TYPE_CONTROL      = 3,
        LLAMA_TOKEN_TYPE_USER_DEFINED = 4,
        LLAMA_TOKEN_TYPE_UNUSED       = 5,
        LLAMA_TOKEN_TYPE_BYTE         = 6,
    };

    enum llama_token_attr {
        LLAMA_TOKEN_ATTR_UNDEFINED    = 0,
        LLAMA_TOKEN_ATTR_UNKNOWN      = 1 << 0,
        LLAMA_TOKEN_ATTR_UNUSED       = 1 << 1,
        LLAMA_TOKEN_ATTR_NORMAL       = 1 << 2,
        LLAMA_TOKEN_ATTR_CONTROL      = 1 << 3,  // SPECIAL?
        LLAMA_TOKEN_ATTR_USER_DEFINED = 1 << 4,
        LLAMA_TOKEN_ATTR_BYTE         = 1 << 5,
        LLAMA_TOKEN_ATTR_NORMALIZED   = 1 << 6,
        LLAMA_TOKEN_ATTR_LSTRIP       = 1 << 7,
        LLAMA_TOKEN_ATTR_RSTRIP       = 1 << 8,
        LLAMA_TOKEN_ATTR_SINGLE_WORD  = 1 << 9,
    };

    // model file types
    enum llama_ftype {
        LLAMA_FTYPE_ALL_F32              = 0,
        LLAMA_FTYPE_MOSTLY_F16           = 1,  // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q4_0          = 2,  // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q4_1          = 3,  // except 1d tensors
        // LLAMA_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4,  // tok_embeddings.weight and output.weight are F16
        // LLAMA_FTYPE_MOSTLY_Q4_2       = 5,  // support has been removed
        // LLAMA_FTYPE_MOSTLY_Q4_3       = 6,  // support has been removed
        LLAMA_FTYPE_MOSTLY_Q8_0          = 7,  // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q5_0          = 8,  // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q5_1          = 9,  // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q2_K          = 10, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q3_K_S        = 11, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q3_K_M        = 12, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q3_K_L        = 13, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q4_K_S        = 14, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q4_K_M        = 15, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q5_K_S        = 16, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q5_K_M        = 17, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q6_K          = 18, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ2_XXS       = 19, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ2_XS        = 20, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q2_K_S        = 21, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ3_XS        = 22, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ3_XXS       = 23, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ1_S         = 24, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ4_NL        = 25, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ3_S         = 26, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ3_M         = 27, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ2_S         = 28, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ2_M         = 29, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ4_XS        = 30, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ1_M         = 31, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_BF16          = 32, // except 1d tensors
        //LLAMA_FTYPE_MOSTLY_Q4_0_4_4      = 33, // removed from gguf files, use Q4_0 and runtime repack
        //LLAMA_FTYPE_MOSTLY_Q4_0_4_8      = 34, // removed from gguf files, use Q4_0 and runtime repack
        //LLAMA_FTYPE_MOSTLY_Q4_0_8_8      = 35, // removed from gguf files, use Q4_0 and runtime repack
        LLAMA_FTYPE_MOSTLY_TQ1_0         = 36, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_TQ2_0         = 37, // except 1d tensors

        LLAMA_FTYPE_GUESSED = 1024, // not specified in the model file
    };

    enum llama_rope_scaling_type {
        LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED = -1,
        LLAMA_ROPE_SCALING_TYPE_NONE        = 0,
        LLAMA_ROPE_SCALING_TYPE_LINEAR      = 1,
        LLAMA_ROPE_SCALING_TYPE_YARN        = 2,
        LLAMA_ROPE_SCALING_TYPE_LONGROPE    = 3,
        LLAMA_ROPE_SCALING_TYPE_MAX_VALUE   = LLAMA_ROPE_SCALING_TYPE_LONGROPE,
    };

    enum llama_pooling_type {
        LLAMA_POOLING_TYPE_UNSPECIFIED = -1,
        LLAMA_POOLING_TYPE_NONE = 0,
        LLAMA_POOLING_TYPE_MEAN = 1,
        LLAMA_POOLING_TYPE_CLS  = 2,
        LLAMA_POOLING_TYPE_LAST = 3,
        LLAMA_POOLING_TYPE_RANK = 4, // used by reranking models to attach the classification head to the graph
    };

    enum llama_attention_type {
        LLAMA_ATTENTION_TYPE_UNSPECIFIED = -1,
        LLAMA_ATTENTION_TYPE_CAUSAL      = 0,
        LLAMA_ATTENTION_TYPE_NON_CAUSAL  = 1,
    };

    enum llama_split_mode {
        LLAMA_SPLIT_MODE_NONE  = 0, // single GPU
        LLAMA_SPLIT_MODE_LAYER = 1, // split layers and KV across GPUs
        LLAMA_SPLIT_MODE_ROW   = 2, // split layers and KV across GPUs, use tensor parallelism if supported
    };

    // TODO: simplify (https://github.com/ggml-org/llama.cpp/pull/9294#pullrequestreview-2286561979)
    typedef struct llama_token_data {
        llama_token id; // token id
        float logit;    // log-odds of the token
        float p;        // probability of the token
    } llama_token_data;

    typedef struct llama_token_data_array {
        // TODO: consider SoA
        // NOTE: this pointer can be modified by the samplers
        llama_token_data * data;
        size_t size;
        int64_t selected; // this is the index in the data array (i.e. not the token id)
        bool sorted;
    } llama_token_data_array;

    typedef bool (*llama_progress_callback)(float progress, void * user_data);

    // Input data for llama_decode
    // A llama_batch object can contain input about one or many sequences
    // The provided arrays (i.e. token, embd, pos, etc.) must have size of n_tokens
    //
    // - token  : the token ids of the input (used when embd is NULL)
    // - embd   : token embeddings (i.e. float vector of size n_embd) (used when token is NULL)
    // - pos    : the positions of the respective token in the sequence
    //            (if set to NULL, the token position will be tracked automatically by llama_decode)
    // - seq_id : the sequence to which the respective token belongs
    //            (if set to NULL, the sequence ID will be assumed to be 0)
    // - logits : if zero, the logits (and/or the embeddings) for the respective token will not be output
    //            (if set to NULL, only the logits for last token will be returned)
    //
    typedef struct llama_batch {
        int32_t n_tokens;

        llama_token  *  token;
        float        *  embd;
        llama_pos    *  pos;
        int32_t      *  n_seq_id;
        llama_seq_id ** seq_id;
        int8_t       *  logits; // TODO: rename this to "output"
    } llama_batch;

    enum llama_model_kv_override_type {
        LLAMA_KV_OVERRIDE_TYPE_INT,
        LLAMA_KV_OVERRIDE_TYPE_FLOAT,
        LLAMA_KV_OVERRIDE_TYPE_BOOL,
        LLAMA_KV_OVERRIDE_TYPE_STR,
    };

    struct llama_model_kv_override {
        enum llama_model_kv_override_type tag;

        char key[128];

        union {
            int64_t val_i64;
            double  val_f64;
            bool    val_bool;
            char    val_str[128];
        };
    };

    struct llama_model_tensor_buft_override {
        const char * pattern;
        ggml_backend_buffer_type_t buft;
    };

    struct llama_model_params {
        // NULL-terminated list of devices to use for offloading (if NULL, all available devices are used)
        ggml_backend_dev_t * devices;

        // NULL-terminated list of buffer types to use for tensors that match a pattern
        const struct llama_model_tensor_buft_override * tensor_buft_overrides;

        int32_t n_gpu_layers; // number of layers to store in VRAM
        enum llama_split_mode split_mode; // how to split the model across multiple GPUs

        // the GPU that is used for the entire model when split_mode is LLAMA_SPLIT_MODE_NONE
        int32_t main_gpu;

        // proportion of the model (layers or rows) to offload to each GPU, size: llama_max_devices()
        const float * tensor_split;

        // Called with a progress value between 0.0 and 1.0. Pass NULL to disable.
        // If the provided progress_callback returns true, model loading continues.
        // If it returns false, model loading is immediately aborted.
        llama_progress_callback progress_callback;

        // context pointer passed to the progress callback
        void * progress_callback_user_data;

        // override key-value pairs of the model meta data
        const struct llama_model_kv_override * kv_overrides;

        // Keep the booleans together to avoid misalignment during copy-by-value.
        bool vocab_only;    // only load the vocabulary, no weights
        bool use_mmap;      // use mmap if possible
        bool use_mlock;     // force system to keep model in RAM
        bool check_tensors; // validate model tensor data
    };

    // NOTE: changing the default values of parameters marked as [EXPERIMENTAL] may cause crashes or incorrect results in certain configurations
    //       https://github.com/ggml-org/llama.cpp/pull/7544
    struct llama_context_params {
        uint32_t n_ctx;             // text context, 0 = from model
        uint32_t n_batch;           // logical maximum batch size that can be submitted to llama_decode
        uint32_t n_ubatch;          // physical maximum batch size
        uint32_t n_seq_max;         // max number of sequences (i.e. distinct states for recurrent models)
        int32_t  n_threads;         // number of threads to use for generation
        int32_t  n_threads_batch;   // number of threads to use for batch processing

        enum llama_rope_scaling_type rope_scaling_type; // RoPE scaling type, from `enum llama_rope_scaling_type`
        enum llama_pooling_type      pooling_type;      // whether to pool (sum) embedding results by sequence id
        enum llama_attention_type    attention_type;    // attention type to use for embeddings

        // ref: https://github.com/ggml-org/llama.cpp/pull/2054
        float    rope_freq_base;   // RoPE base frequency, 0 = from model
        float    rope_freq_scale;  // RoPE frequency scaling factor, 0 = from model
        float    yarn_ext_factor;  // YaRN extrapolation mix factor, negative = from model
        float    yarn_attn_factor; // YaRN magnitude scaling factor
        float    yarn_beta_fast;   // YaRN low correction dim
        float    yarn_beta_slow;   // YaRN high correction dim
        uint32_t yarn_orig_ctx;    // YaRN original context size
        float    defrag_thold;     // defragment the KV cache if holes/size > thold, <= 0 disabled (default)

        ggml_backend_sched_eval_callback cb_eval;
        void * cb_eval_user_data;

        enum ggml_type type_k; // data type for K cache [EXPERIMENTAL]
        enum ggml_type type_v; // data type for V cache [EXPERIMENTAL]

        // Abort callback
        // if it returns true, execution of llama_decode() will be aborted
        // currently works only with CPU execution
        ggml_abort_callback abort_callback;
        void *              abort_callback_data;

        // Keep the booleans together and at the end of the struct to avoid misalignment during copy-by-value.
        bool embeddings;  // if true, extract embeddings (together with logits)
        bool offload_kqv; // offload the KQV ops (including the KV cache) to GPU
        bool flash_attn;  // use flash attention [EXPERIMENTAL]
        bool no_perf;     // measure performance timings
        bool op_offload;  // offload host tensor operations to device
        bool swa_full;    // use full-size SWA cache (https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055)
    };

    // model quantization parameters
    typedef struct llama_model_quantize_params {
        int32_t nthread;                      // number of threads to use for quantizing, if <=0 will use std::thread::hardware_concurrency()
        enum llama_ftype ftype;               // quantize to this llama_ftype
        enum ggml_type output_tensor_type;    // output tensor type
        enum ggml_type token_embedding_type;  // token embeddings tensor type
        bool allow_requantize;                // allow quantizing non-f32/f16 tensors
        bool quantize_output_tensor;          // quantize output.weight
        bool only_copy;                       // only copy tensors - ftype, allow_requantize and quantize_output_tensor are ignored
        bool pure;                            // quantize all tensors to the default type
        bool keep_split;                      // quantize to the same number of shards
        void * imatrix;                       // pointer to importance matrix data
        void * kv_overrides;                  // pointer to vector containing overrides
        void * tensor_types;                  // pointer to vector containing tensor types
    } llama_model_quantize_params;

    typedef struct llama_logit_bias {
        llama_token token;
        float bias;
    } llama_logit_bias;

    typedef struct llama_sampler_chain_params {
        bool no_perf; // whether to measure performance timings
    } llama_sampler_chain_params;

    // used in chat template
    typedef struct llama_chat_message {
        const char * role;
        const char * content;
    } llama_chat_message;

    // lora adapter
    struct llama_adapter_lora;

    // Helpers for getting default parameters
    // TODO: update API to start accepting pointers to params structs (https://github.com/ggml-org/llama.cpp/discussions/9172)
    LLAMA_API struct llama_model_params          llama_model_default_params(void);
    LLAMA_API struct llama_context_params        llama_context_default_params(void);
    LLAMA_API struct llama_sampler_chain_params  llama_sampler_chain_default_params(void);
    LLAMA_API struct llama_model_quantize_params llama_model_quantize_default_params(void);

    // Initialize the llama + ggml backend
    // If numa is true, use NUMA optimizations
    // Call once at the start of the program
    LLAMA_API void llama_backend_init(void);

    // Call once at the end of the program - currently only used for MPI
    LLAMA_API void llama_backend_free(void);

    //optional:
    LLAMA_API void llama_numa_init(enum ggml_numa_strategy numa);

    // Optional: an auto threadpool gets created in ggml if not passed explicitly
    LLAMA_API void llama_attach_threadpool(
            struct llama_context * ctx,
               ggml_threadpool_t   threadpool,
               ggml_threadpool_t   threadpool_batch);

    LLAMA_API void llama_detach_threadpool(struct llama_context * ctx);

    DEPRECATED(LLAMA_API struct llama_model * llama_load_model_from_file(
                             const char * path_model,
              struct llama_model_params   params),
            "use llama_model_load_from_file instead");

    // Load the model from a file
    // If the file is split into multiple parts, the file name must follow this pattern: <name>-%05d-of-%05d.gguf
    // If the split file name does not follow this pattern, use llama_model_load_from_splits
    LLAMA_API struct llama_model * llama_model_load_from_file(
                             const char * path_model,
              struct llama_model_params   params);

    // Load the model from multiple splits (support custom naming scheme)
    // The paths must be in the correct order
    LLAMA_API struct llama_model * llama_model_load_from_splits(
                             const char ** paths,
                                 size_t    n_paths,
              struct llama_model_params    params);

    LLAMA_API void llama_model_save_to_file(
            const struct llama_model * model,
                        const char * path_model);

    DEPRECATED(LLAMA_API void llama_free_model(struct llama_model * model),
            "use llama_model_free instead");

    LLAMA_API void llama_model_free(struct llama_model * model);

    LLAMA_API struct llama_context * llama_init_from_model(
                     struct llama_model * model,
            struct llama_context_params   params);

    DEPRECATED(LLAMA_API struct llama_context * llama_new_context_with_model(
                     struct llama_model * model,
            struct llama_context_params   params),
            "use llama_init_from_model instead");

    // Frees all allocated memory
    LLAMA_API void llama_free(struct llama_context * ctx);

    LLAMA_API int64_t llama_time_us(void);

    LLAMA_API size_t llama_max_devices(void);

    LLAMA_API bool llama_supports_mmap       (void);
    LLAMA_API bool llama_supports_mlock      (void);
    LLAMA_API bool llama_supports_gpu_offload(void);
    LLAMA_API bool llama_supports_rpc        (void);

    LLAMA_API uint32_t llama_n_ctx      (const struct llama_context * ctx);
    LLAMA_API uint32_t llama_n_batch    (const struct llama_context * ctx);
    LLAMA_API uint32_t llama_n_ubatch   (const struct llama_context * ctx);
    LLAMA_API uint32_t llama_n_seq_max  (const struct llama_context * ctx);

    DEPRECATED(LLAMA_API int32_t llama_n_ctx_train(const struct llama_model * model), "use llama_model_n_ctx_train instead");
    DEPRECATED(LLAMA_API int32_t llama_n_embd     (const struct llama_model * model), "use llama_model_n_embd instead");
    DEPRECATED(LLAMA_API int32_t llama_n_layer    (const struct llama_model * model), "use llama_model_n_layer instead");
    DEPRECATED(LLAMA_API int32_t llama_n_head     (const struct llama_model * model), "use llama_model_n_head instead");

    DEPRECATED(LLAMA_API int32_t llama_n_vocab    (const struct llama_vocab * vocab), "use llama_vocab_n_tokens instead");

    LLAMA_API const struct llama_model * llama_get_model   (const struct llama_context * ctx);
    LLAMA_API    struct llama_kv_cache * llama_get_kv_self (      struct llama_context * ctx);
    LLAMA_API  enum llama_pooling_type   llama_pooling_type(const struct llama_context * ctx); // TODO: rename to llama_get_pooling_type

    LLAMA_API const struct llama_vocab * llama_model_get_vocab(const struct llama_model * model);
    LLAMA_API enum llama_rope_type       llama_model_rope_type(const struct llama_model * model);

    LLAMA_API int32_t llama_model_n_ctx_train(const struct llama_model * model);
    LLAMA_API int32_t llama_model_n_embd     (const struct llama_model * model);
    LLAMA_API int32_t llama_model_n_layer    (const struct llama_model * model);
    LLAMA_API int32_t llama_model_n_head     (const struct llama_model * model);
    LLAMA_API int32_t llama_model_n_head_kv  (const struct llama_model * model);

    // Get the model's RoPE frequency scaling factor
    LLAMA_API float llama_model_rope_freq_scale_train(const struct llama_model * model);

    LLAMA_API enum llama_vocab_type llama_vocab_type(const struct llama_vocab * vocab);

    LLAMA_API int32_t llama_vocab_n_tokens(const struct llama_vocab * vocab);

    // Functions to access the model's GGUF metadata scalar values
    // - The functions return the length of the string on success, or -1 on failure
    // - The output string is always null-terminated and cleared on failure
    // - When retrieving a string, an extra byte must be allocated to account for the null terminator
    // - GGUF array values are not supported by these functions

    // Get metadata value as a string by key name
    LLAMA_API int32_t llama_model_meta_val_str(const struct llama_model * model, const char * key, char * buf, size_t buf_size);

    // Get the number of metadata key/value pairs
    LLAMA_API int32_t llama_model_meta_count(const struct llama_model * model);

    // Get metadata key name by index
    LLAMA_API int32_t llama_model_meta_key_by_index(const struct llama_model * model, int32_t i, char * buf, size_t buf_size);

    // Get metadata value as a string by index
    LLAMA_API int32_t llama_model_meta_val_str_by_index(const struct llama_model * model, int32_t i, char * buf, size_t buf_size);

    // Get a string describing the model type
    LLAMA_API int32_t llama_model_desc(const struct llama_model * model, char * buf, size_t buf_size);

    // Returns the total size of all the tensors in the model in bytes
    LLAMA_API uint64_t llama_model_size(const struct llama_model * model);

    // Get the default chat template. Returns nullptr if not available
    // If name is NULL, returns the default chat template
    LLAMA_API const char * llama_model_chat_template(const struct llama_model * model, const char * name);

    // Returns the total number of parameters in the model
    LLAMA_API uint64_t llama_model_n_params(const struct llama_model * model);

    // Returns true if the model contains an encoder that requires llama_encode() call
    LLAMA_API bool llama_model_has_encoder(const struct llama_model * model);

    // Returns true if the model contains a decoder that requires llama_decode() call
    LLAMA_API bool llama_model_has_decoder(const struct llama_model * model);

    // For encoder-decoder models, this function returns id of the token that must be provided
    // to the decoder to start generating output sequence. For other models, it returns -1.
    LLAMA_API llama_token llama_model_decoder_start_token(const struct llama_model * model);

    // Returns true if the model is recurrent (like Mamba, RWKV, etc.)
    LLAMA_API bool llama_model_is_recurrent(const struct llama_model * model);

    // Returns 0 on success
    LLAMA_API uint32_t llama_model_quantize(
            const char * fname_inp,
            const char * fname_out,
            const llama_model_quantize_params * params);

    //
    // Adapters
    //

    // Load a LoRA adapter from file
    LLAMA_API struct llama_adapter_lora * llama_adapter_lora_init(
            struct llama_model * model,
            const char * path_lora);

    // Manually free a LoRA adapter
    // Note: loaded adapters will be free when the associated model is deleted
    LLAMA_API void llama_adapter_lora_free(struct llama_adapter_lora * adapter);

    // The following functions operate on a llama_context, hence the naming: llama_verb_...

    // Add a loaded LoRA adapter to given context
    // This will not modify model's weight
    LLAMA_API int32_t llama_set_adapter_lora(
            struct llama_context * ctx,
            struct llama_adapter_lora * adapter,
            float scale);

    // Remove a specific LoRA adapter from given context
    // Return -1 if the adapter is not present in the context
    LLAMA_API int32_t llama_rm_adapter_lora(
            struct llama_context * ctx,
            struct llama_adapter_lora * adapter);

    // Remove all LoRA adapters from given context
    LLAMA_API void llama_clear_adapter_lora(struct llama_context * ctx);

    // Apply a loaded control vector to a llama_context, or if data is NULL, clear
    // the currently loaded vector.
    // n_embd should be the size of a single layer's control, and data should point
    // to an n_embd x n_layers buffer starting from layer 1.
    // il_start and il_end are the layer range the vector should apply to (both inclusive)
    // See llama_control_vector_load in common to load a control vector.
    LLAMA_API int32_t llama_apply_adapter_cvec(
            struct llama_context * ctx,
                     const float * data,
                          size_t   len,
                         int32_t   n_embd,
                         int32_t   il_start,
                         int32_t   il_end);

    //
    // KV cache
    //

    // Returns the number of tokens in the KV cache (slow, use only for debug)
    // If a KV cell has multiple sequences assigned to it, it will be counted multiple times
    LLAMA_API int32_t llama_kv_self_n_tokens(const struct llama_context * ctx);

    // Returns the number of used KV cells (i.e. have at least one sequence assigned to them)
    LLAMA_API int32_t llama_kv_self_used_cells(const struct llama_context * ctx);

    // Clear the KV cache - both cell info is erased and KV data is zeroed
    LLAMA_API void llama_kv_self_clear(
            struct llama_context * ctx);

    // Removes all tokens that belong to the specified sequence and have positions in [p0, p1)
    // Returns false if a partial sequence cannot be removed. Removing a whole sequence never fails
    // seq_id < 0 : match any sequence
    // p0 < 0     : [0,  p1]
    // p1 < 0     : [p0, inf)
    LLAMA_API bool llama_kv_self_seq_rm(
            struct llama_context * ctx,
                    llama_seq_id   seq_id,
                       llama_pos   p0,
                       llama_pos   p1);

    // Copy all tokens that belong to the specified sequence to another sequence
    // Note that this does not allocate extra KV cache memory - it simply assigns the tokens to the new sequence
    // p0 < 0 : [0,  p1]
    // p1 < 0 : [p0, inf)
    LLAMA_API void llama_kv_self_seq_cp(
            struct llama_context * ctx,
                    llama_seq_id   seq_id_src,
                    llama_seq_id   seq_id_dst,
                       llama_pos   p0,
                       llama_pos   p1);

    // Removes all tokens that do not belong to the specified sequence
    LLAMA_API void llama_kv_self_seq_keep(
            struct llama_context * ctx,
                    llama_seq_id   seq_id);

    // Adds relative position "delta" to all tokens that belong to the specified sequence and have positions in [p0, p1)
    // If the KV cache is RoPEd, the KV data is updated accordingly:
    //   - lazily on next llama_decode()
    //   - explicitly with llama_kv_self_update()
    // p0 < 0 : [0,  p1]
    // p1 < 0 : [p0, inf)
    LLAMA_API void llama_kv_self_seq_add(
            struct llama_context * ctx,
                    llama_seq_id   seq_id,
                       llama_pos   p0,
                       llama_pos   p1,
                       llama_pos   delta);

    // Integer division of the positions by factor of `d > 1`
    // If the KV cache is RoPEd, the KV data is updated accordingly:
    //   - lazily on next llama_decode()
    //   - explicitly with llama_kv_self_update()
    // p0 < 0 : [0,  p1]
    // p1 < 0 : [p0, inf)
    LLAMA_API void llama_kv_self_seq_div(
            struct llama_context * ctx,
                    llama_seq_id   seq_id,
                       llama_pos   p0,
                       llama_pos   p1,
                             int   d);

    // Returns the smallest position present in the KV cache for the specified sequence
    // This is typically non-zero only for SWA caches
    // Return -1 if the sequence is empty
    LLAMA_API llama_pos llama_kv_self_seq_pos_min(
            struct llama_context * ctx,
                    llama_seq_id   seq_id);

    // Returns the largest position present in the KV cache for the specified sequence
    // Return -1 if the sequence is empty
    LLAMA_API llama_pos llama_kv_self_seq_pos_max(
            struct llama_context * ctx,
                    llama_seq_id   seq_id);

    // Defragment the KV cache
    // This will be applied:
    //   - lazily on next llama_decode()
    //   - explicitly with llama_kv_self_update()
    LLAMA_API void llama_kv_self_defrag(struct llama_context * ctx);

    // Check if the context supports KV cache shifting
    LLAMA_API bool llama_kv_self_can_shift(const struct llama_context * ctx);

    // Apply the KV cache updates (such as K-shifts, defragmentation, etc.)
    LLAMA_API void llama_kv_self_update(struct llama_context * ctx);

    //
    // State / sessions
    //

    // Returns the *actual* size in bytes of the state
    // (logits, embedding and kv_cache)
    // Only use when saving the state, not when restoring it, otherwise the size may be too small.
    LLAMA_API size_t llama_state_get_size(struct llama_context * ctx);
    LLAMA_API DEPRECATED(size_t llama_get_state_size(struct llama_context * ctx),
        "use llama_state_get_size instead");

    // Copies the state to the specified destination address.
    // Destination needs to have allocated enough memory.
    // Returns the number of bytes copied
    LLAMA_API size_t llama_state_get_data(
            struct llama_context * ctx,
                         uint8_t * dst,
                          size_t   size);
    LLAMA_API DEPRECATED(size_t llama_copy_state_data(
            struct llama_context * ctx,
                         uint8_t * dst),
        "use llama_state_get_data instead");

    // Set the state reading from the specified address
    // Returns the number of bytes read
    LLAMA_API size_t llama_state_set_data(
            struct llama_context * ctx,
                   const uint8_t * src,
                          size_t   size);
    LLAMA_API DEPRECATED(size_t llama_set_state_data(
            struct llama_context * ctx,
                   const uint8_t * src),
        "use llama_state_set_data instead");

    // Save/load session file
    LLAMA_API bool llama_state_load_file(
            struct llama_context * ctx,
                      const char * path_session,
                     llama_token * tokens_out,
                          size_t   n_token_capacity,
                          size_t * n_token_count_out);
    LLAMA_API DEPRECATED(bool llama_load_session_file(
            struct llama_context * ctx,
                      const char * path_session,
                     llama_token * tokens_out,
                          size_t   n_token_capacity,
                          size_t * n_token_count_out),
        "use llama_state_load_file instead");

    LLAMA_API bool llama_state_save_file(
            struct llama_context * ctx,
                      const char * path_session,
               const llama_token * tokens,
                          size_t   n_token_count);
    LLAMA_API DEPRECATED(bool llama_save_session_file(
            struct llama_context * ctx,
                      const char * path_session,
               const llama_token * tokens,
                          size_t   n_token_count),
        "use llama_state_save_file instead");

    // Get the exact size needed to copy the KV cache of a single sequence
    LLAMA_API size_t llama_state_seq_get_size(
            struct llama_context * ctx,
                    llama_seq_id   seq_id);

    // Copy the KV cache of a single sequence into the specified buffer
    LLAMA_API size_t llama_state_seq_get_data(
            struct llama_context * ctx,
                         uint8_t * dst,
                          size_t   size,
                    llama_seq_id   seq_id);

    // Copy the sequence data (originally copied with `llama_state_seq_get_data`) into the specified sequence
    // Returns:
    //  - Positive: Ok
    //  - Zero: Failed to load
    LLAMA_API size_t llama_state_seq_set_data(
            struct llama_context * ctx,
                   const uint8_t * src,
                          size_t   size,
                    llama_seq_id   dest_seq_id);

    LLAMA_API size_t llama_state_seq_save_file(
            struct llama_context * ctx,
                      const char * filepath,
                    llama_seq_id   seq_id,
               const llama_token * tokens,
                          size_t   n_token_count);

    LLAMA_API size_t llama_state_seq_load_file(
            struct llama_context * ctx,
                      const char * filepath,
                    llama_seq_id   dest_seq_id,
                     llama_token * tokens_out,
                          size_t   n_token_capacity,
                          size_t * n_token_count_out);

    //
    // Decoding
    //

    // Return batch for single sequence of tokens
    // The sequence ID will be fixed to 0
    // The position of the tokens will be tracked automatically by llama_decode
    //
    // NOTE: this is a helper function to facilitate transition to the new batch API - avoid using it
    //
    LLAMA_API struct llama_batch llama_batch_get_one(
                  llama_token * tokens,
                      int32_t   n_tokens);

    // Allocates a batch of tokens on the heap that can hold a maximum of n_tokens
    // Each token can be assigned up to n_seq_max sequence ids
    // The batch has to be freed with llama_batch_free()
    // If embd != 0, llama_batch.embd will be allocated with size of n_tokens * embd * sizeof(float)
    // Otherwise, llama_batch.token will be allocated to store n_tokens llama_token
    // The rest of the llama_batch members are allocated with size n_tokens
    // All members are left uninitialized
    LLAMA_API struct llama_batch llama_batch_init(
            int32_t n_tokens,
            int32_t embd,
            int32_t n_seq_max);

    // Frees a batch of tokens allocated with llama_batch_init()
    LLAMA_API void llama_batch_free(struct llama_batch batch);

    // Process a batch of tokens.
    // In contrast to llama_decode() - this call does not use KV cache.
    // For encode-decoder contexts, processes the batch using the encoder.
    // Can store the encoder output internally for later use by the decoder's cross-attention layers.
    //   0 - success
    // < 0 - error. the KV cache state is restored to the state before this call
    LLAMA_API int32_t llama_encode(
            struct llama_context * ctx,
              struct llama_batch   batch);

    // Process a batch of tokens.
    // Requires KV cache.
    // For encode-decoder contexts, processes the batch using the decoder.
    // Positive return values does not mean a fatal error, but rather a warning.
    // Upon non-zero return values, the KV cache state is restored to the state before this call
    //    0 - success
    //    1 - could not find a KV slot for the batch (try reducing the size of the batch or increase the context)
    //    2 - aborted
    //   -1 - invalid input batch
    // < -1 - error
    LLAMA_API int32_t llama_decode(
            struct llama_context * ctx,
              struct llama_batch   batch);

    // Set the number of threads used for decoding
    // n_threads is the number of threads used for generation (single token)
    // n_threads_batch is the number of threads used for prompt and batch processing (multiple tokens)
    LLAMA_API void llama_set_n_threads(struct llama_context * ctx, int32_t n_threads, int32_t n_threads_batch);

    // Get the number of threads used for generation of a single token.
    LLAMA_API int32_t llama_n_threads(struct llama_context * ctx);

    // Get the number of threads used for prompt and batch processing (multiple token).
    LLAMA_API int32_t llama_n_threads_batch(struct llama_context * ctx);

    // Set whether the model is in embeddings mode or not
    // If true, embeddings will be returned but logits will not
    LLAMA_API void llama_set_embeddings(struct llama_context * ctx, bool embeddings);

    // Set whether to use causal attention or not
    // If set to true, the model will only attend to the past tokens
    LLAMA_API void llama_set_causal_attn(struct llama_context * ctx, bool causal_attn);

    // Set whether the model is in warmup mode or not
    // If true, all model tensors are activated during llama_decode() to load and cache their weights.
    LLAMA_API void llama_set_warmup(struct llama_context * ctx, bool warmup);

    // Set abort callback
    LLAMA_API void llama_set_abort_callback(struct llama_context * ctx, ggml_abort_callback abort_callback, void * abort_callback_data);

    // Wait until all computations are finished
    // This is automatically done when using one of the functions below to obtain the computation results
    // and is not necessary to call it explicitly in most cases
    LLAMA_API void llama_synchronize(struct llama_context * ctx);

    // Token logits obtained from the last call to llama_decode()
    // The logits for which llama_batch.logits[i] != 0 are stored contiguously
    // in the order they have appeared in the batch.
    // Rows: number of tokens for which llama_batch.logits[i] != 0
    // Cols: n_vocab
    LLAMA_API float * llama_get_logits(struct llama_context * ctx);

    // Logits for the ith token. For positive indices, Equivalent to:
    // llama_get_logits(ctx) + ctx->output_ids[i]*n_vocab
    // Negative indicies can be used to access logits in reverse order, -1 is the last logit.
    // returns NULL for invalid ids.
    LLAMA_API float * llama_get_logits_ith(struct llama_context * ctx, int32_t i);

    // Get all output token embeddings.
    // when pooling_type == LLAMA_POOLING_TYPE_NONE or when using a generative model,
    // the embeddings for which llama_batch.logits[i] != 0 are stored contiguously
    // in the order they have appeared in the batch.
    // shape: [n_outputs*n_embd]
    // Otherwise, returns NULL.
    LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);

    // Get the embeddings for the ith token. For positive indices, Equivalent to:
    // llama_get_embeddings(ctx) + ctx->output_ids[i]*n_embd
    // Negative indicies can be used to access embeddings in reverse order, -1 is the last embedding.
    // shape: [n_embd] (1-dimensional)
    // returns NULL for invalid ids.
    LLAMA_API float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i);

    // Get the embeddings for a sequence id
    // Returns NULL if pooling_type is LLAMA_POOLING_TYPE_NONE
    // when pooling_type == LLAMA_POOLING_TYPE_RANK, returns float[1] with the rank of the sequence
    // otherwise: float[n_embd] (1-dimensional)
    LLAMA_API float * llama_get_embeddings_seq(struct llama_context * ctx, llama_seq_id seq_id);

    //
    // Vocab
    //

    LLAMA_API const char * llama_vocab_get_text(const struct llama_vocab * vocab, llama_token token);

    LLAMA_API float llama_vocab_get_score(const struct llama_vocab * vocab, llama_token token);

    LLAMA_API enum llama_token_attr llama_vocab_get_attr(const struct llama_vocab * vocab, llama_token token);

    // Check if the token is supposed to end generation (end-of-generation, eg. EOS, EOT, etc.)
    LLAMA_API bool llama_vocab_is_eog(const struct llama_vocab * vocab, llama_token token);

    // Identify if Token Id is a control token or a render-able token
    LLAMA_API bool llama_vocab_is_control(const struct llama_vocab * vocab, llama_token token);

    // Special tokens
    LLAMA_API llama_token llama_vocab_bos(const struct llama_vocab * vocab); // beginning-of-sentence
    LLAMA_API llama_token llama_vocab_eos(const struct llama_vocab * vocab); // end-of-sentence
    LLAMA_API llama_token llama_vocab_eot(const struct llama_vocab * vocab); // end-of-turn
    LLAMA_API llama_token llama_vocab_sep(const struct llama_vocab * vocab); // sentence separator
    LLAMA_API llama_token llama_vocab_nl (const struct llama_vocab * vocab); // next-line
    LLAMA_API llama_token llama_vocab_pad(const struct llama_vocab * vocab); // padding

    LLAMA_API bool llama_vocab_get_add_bos(const struct llama_vocab * vocab);
    LLAMA_API bool llama_vocab_get_add_eos(const struct llama_vocab * vocab);

    LLAMA_API llama_token llama_vocab_fim_pre(const struct llama_vocab * vocab);
    LLAMA_API llama_token llama_vocab_fim_suf(const struct llama_vocab * vocab);
    LLAMA_API llama_token llama_vocab_fim_mid(const struct llama_vocab * vocab);
    LLAMA_API llama_token llama_vocab_fim_pad(const struct llama_vocab * vocab);
    LLAMA_API llama_token llama_vocab_fim_rep(const struct llama_vocab * vocab);
    LLAMA_API llama_token llama_vocab_fim_sep(const struct llama_vocab * vocab);

    DEPRECATED(LLAMA_API const char * llama_token_get_text(const struct llama_vocab * vocab, llama_token token), "use llama_vocab_get_text instead");
    DEPRECATED(LLAMA_API float llama_token_get_score(const struct llama_vocab * vocab, llama_token token), "use llama_vocab_get_score instead");
    DEPRECATED(LLAMA_API enum llama_token_attr llama_token_get_attr(const struct llama_vocab * vocab, llama_token token), "use llama_vocab_get_attr instead");
    DEPRECATED(LLAMA_API bool llama_token_is_eog(const struct llama_vocab * vocab, llama_token token), "use llama_vocab_is_eog instead");
    DEPRECATED(LLAMA_API bool llama_token_is_control(const struct llama_vocab * vocab, llama_token token), "use llama_vocab_is_control instead");
    DEPRECATED(LLAMA_API llama_token llama_token_bos(const struct llama_vocab * vocab), "use llama_vocab_bos instead");
    DEPRECATED(LLAMA_API llama_token llama_token_eos(const struct llama_vocab * vocab), "use llama_vocab_eos instead");
    DEPRECATED(LLAMA_API llama_token llama_token_eot(const struct llama_vocab * vocab), "use llama_vocab_eot instead");
    DEPRECATED(LLAMA_API llama_token llama_token_cls(const struct llama_vocab * vocab), "use llama_vocab_cls instead");
    DEPRECATED(LLAMA_API llama_token llama_token_sep(const struct llama_vocab * vocab), "use llama_vocab_sep instead");
    DEPRECATED(LLAMA_API llama_token llama_token_nl (const struct llama_vocab * vocab), "use llama_vocab_nl instead");
    DEPRECATED(LLAMA_API llama_token llama_token_pad(const struct llama_vocab * vocab), "use llama_vocab_pad instead");
    DEPRECATED(LLAMA_API bool llama_add_bos_token(const struct llama_vocab * vocab), "use llama_vocab_get_add_bos instead");
    DEPRECATED(LLAMA_API bool llama_add_eos_token(const struct llama_vocab * vocab), "use llama_vocab_get_add_eos instead");
    DEPRECATED(LLAMA_API llama_token llama_token_fim_pre(const struct llama_vocab * vocab), "use llama_vocab_fim_pre instead");
    DEPRECATED(LLAMA_API llama_token llama_token_fim_suf(const struct llama_vocab * vocab), "use llama_vocab_fim_suf instead");
    DEPRECATED(LLAMA_API llama_token llama_token_fim_mid(const struct llama_vocab * vocab), "use llama_vocab_fim_mid instead");
    DEPRECATED(LLAMA_API llama_token llama_token_fim_pad(const struct llama_vocab * vocab), "use llama_vocab_fim_pad instead");
    DEPRECATED(LLAMA_API llama_token llama_token_fim_rep(const struct llama_vocab * vocab), "use llama_vocab_fim_rep instead");
    DEPRECATED(LLAMA_API llama_token llama_token_fim_sep(const struct llama_vocab * vocab), "use llama_vocab_fim_sep instead");

    // CLS is equivalent to BOS
    DEPRECATED(LLAMA_API llama_token llama_vocab_cls(const struct llama_vocab * vocab), // classification
            "use llama_vocab_bos instead");

    //
    // Tokenization
    //
    // The API is thread-safe.
    //

    /// @details Convert the provided text into tokens.
    /// @param tokens The tokens pointer must be large enough to hold the resulting tokens.
    /// @return Returns the number of tokens on success, no more than n_tokens_max
    /// @return Returns a negative number on failure - the number of tokens that would have been returned
    /// @param add_special Allow to add BOS and EOS tokens if model is configured to do so.
    /// @param parse_special Allow tokenizing special and/or control tokens which otherwise are not exposed and treated
    ///                      as plaintext. Does not insert a leading space.
    LLAMA_API int32_t llama_tokenize(
        const struct llama_vocab * vocab,
                      const char * text,
                         int32_t   text_len,
                     llama_token * tokens,
                         int32_t   n_tokens_max,
                            bool   add_special,
                            bool   parse_special);

    // Token Id -> Piece.
    // Uses the vocabulary in the provided context.
    // Does not write null terminator to the buffer.
    // User can skip up to 'lstrip' leading spaces before copying (useful when encoding/decoding multiple tokens with 'add_space_prefix')
    // @param special If true, special tokens are rendered in the output.
    LLAMA_API int32_t llama_token_to_piece(
              const struct llama_vocab * vocab,
                           llama_token   token,
                                  char * buf,
                               int32_t   length,
                               int32_t   lstrip,
                                  bool   special);

    /// @details Convert the provided tokens into text (inverse of llama_tokenize()).
    /// @param text The char pointer must be large enough to hold the resulting text.
    /// @return Returns the number of chars/bytes on success, no more than text_len_max.
    /// @return Returns a negative number on failure - the number of chars/bytes that would have been returned.
    /// @param remove_special Allow to remove BOS and EOS tokens if model is configured to do so.
    /// @param unparse_special If true, special tokens are rendered in the output.
    LLAMA_API int32_t llama_detokenize(
        const struct llama_vocab * vocab,
               const llama_token * tokens,
                         int32_t   n_tokens,
                            char * text,
                         int32_t   text_len_max,
                            bool   remove_special,
                            bool   unparse_special);

    //
    // Chat templates
    //

    /// Apply chat template. Inspired by hf apply_chat_template() on python.
    /// Both "model" and "custom_template" are optional, but at least one is required. "custom_template" has higher precedence than "model"
    /// NOTE: This function does not use a jinja parser. It only support a pre-defined list of template. See more: https://github.com/ggml-org/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template
    /// @param tmpl A Jinja template to use for this chat. If this is nullptr, the model's default chat template will be used instead.
    /// @param chat Pointer to a list of multiple llama_chat_message
    /// @param n_msg Number of llama_chat_message in this chat
    /// @param add_ass Whether to end the prompt with the token(s) that indicate the start of an assistant message.
    /// @param buf A buffer to hold the output formatted prompt. The recommended alloc size is 2 * (total number of characters of all messages)
    /// @param length The size of the allocated buffer
    /// @return The total number of bytes of the formatted prompt. If is it larger than the size of buffer, you may need to re-alloc it and then re-apply the template.
    LLAMA_API int32_t llama_chat_apply_template(
                            const char * tmpl,
       const struct llama_chat_message * chat,
                                size_t   n_msg,
                                  bool   add_ass,
                                  char * buf,
                               int32_t   length);

    // Get list of built-in chat templates
    LLAMA_API int32_t llama_chat_builtin_templates(const char ** output, size_t len);

    //
    // Sampling API
    //
    // Sample usage:
    //
    //    // prepare the sampling chain at the start
    //    auto sparams = llama_sampler_chain_default_params();
    //
    //    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    //
    //    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(50));
    //    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.9, 1));
    //    llama_sampler_chain_add(smpl, llama_sampler_init_temp (0.8));
    //
    //    // typically, the chain should end with a sampler such as "greedy", "dist" or "mirostat"
    //    // this sampler will be responsible to select the actual token
    //    llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));
    //
    //    ...
    //
    //    // decoding loop:
    //    while (...) {
    //        ...
    //
    //        llama_decode(ctx, batch);
    //
    //        // sample from the logits of the last token in the batch
    //        const llama_token id = llama_sampler_sample(smpl, ctx, -1);
    //
    //        // accepting the token updates the internal state of certain samplers (e.g. grammar, repetition, etc.)
    //        llama_sampler_accept(smpl, id);
    //        ...
    //    }
    //
    //    llama_sampler_free(smpl);
    //
    // TODO: In the future, llama_sampler will be utilized to offload the sampling to the backends (e.g. GPU).
    //

    typedef void * llama_sampler_context_t;

    // user code can implement the interface below in order to create custom llama_sampler
    struct llama_sampler_i {
        const char *           (*name)  (const struct llama_sampler * smpl);                                 // can be NULL
        void                   (*accept)(      struct llama_sampler * smpl, llama_token token);              // can be NULL
        void                   (*apply) (      struct llama_sampler * smpl, llama_token_data_array * cur_p); // required
        void                   (*reset) (      struct llama_sampler * smpl);                                 // can be NULL
        struct llama_sampler * (*clone) (const struct llama_sampler * smpl);                                 // can be NULL if ctx is NULL
        void                   (*free)  (      struct llama_sampler * smpl);                                 // can be NULL if ctx is NULL

        // TODO: API for internal libllama usage for appending the sampling to an existing ggml_cgraph
        //void (*apply_ggml) (struct llama_sampler * smpl, ...);
    };

    struct llama_sampler {
        const struct llama_sampler_i * iface;
        llama_sampler_context_t        ctx;
    };

    // mirror of llama_sampler_i:
    LLAMA_API struct llama_sampler * llama_sampler_init  (const struct llama_sampler_i * iface, llama_sampler_context_t ctx);
    LLAMA_API const char *           llama_sampler_name  (const struct llama_sampler * smpl);
    LLAMA_API void                   llama_sampler_accept(      struct llama_sampler * smpl, llama_token token);
    LLAMA_API void                   llama_sampler_apply (      struct llama_sampler * smpl, llama_token_data_array * cur_p);
    LLAMA_API void                   llama_sampler_reset (      struct llama_sampler * smpl);
    LLAMA_API struct llama_sampler * llama_sampler_clone (const struct llama_sampler * smpl);
    // important: do not free if the sampler has been added to a llama_sampler_chain (via llama_sampler_chain_add)
    LLAMA_API void                   llama_sampler_free  (      struct llama_sampler * smpl);

    // llama_sampler_chain
    // a type of llama_sampler that can chain multiple samplers one after another

    LLAMA_API struct llama_sampler * llama_sampler_chain_init(struct llama_sampler_chain_params params);

    // important: takes ownership of the sampler object and will free it when llama_sampler_free is called
    LLAMA_API void                   llama_sampler_chain_add(      struct llama_sampler * chain, struct llama_sampler * smpl);
    LLAMA_API struct llama_sampler * llama_sampler_chain_get(const struct llama_sampler * chain, int32_t i);
    LLAMA_API int                    llama_sampler_chain_n  (const struct llama_sampler * chain);

    // after removing a sampler, the chain will no longer own it, and it will not be freed when the chain is freed
    LLAMA_API struct llama_sampler * llama_sampler_chain_remove(   struct llama_sampler * chain, int32_t i);

    // available samplers:

    LLAMA_API struct llama_sampler * llama_sampler_init_greedy(void);
    LLAMA_API struct llama_sampler * llama_sampler_init_dist  (uint32_t seed);

    /// @details Sorts candidate tokens by their logits in descending order and calculate probabilities based on logits.
    /// NOTE: Avoid using on the full vocabulary as the sorting can become slow. For example, apply top-k or top-p sampling first.
    DEPRECATED(LLAMA_API struct llama_sampler * llama_sampler_init_softmax    (void),
        "will be removed in the future (see https://github.com/ggml-org/llama.cpp/pull/9896#discussion_r1800920915)");

    /// @details Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
    /// Setting k <= 0 makes this a noop
    LLAMA_API struct llama_sampler * llama_sampler_init_top_k      (int32_t k);

    /// @details Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
    LLAMA_API struct llama_sampler * llama_sampler_init_top_p      (float   p, size_t min_keep);

    /// @details Minimum P sampling as described in https://github.com/ggml-org/llama.cpp/pull/3841
    LLAMA_API struct llama_sampler * llama_sampler_init_min_p      (float   p, size_t min_keep);

    /// @details Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666.
    LLAMA_API struct llama_sampler * llama_sampler_init_typical    (float   p, size_t min_keep);

    /// #details Updates the logits l_i` = l_i/t. When t <= 0.0f, the maximum logit is kept at it's original value, the rest are set to -inf
    LLAMA_API struct llama_sampler * llama_sampler_init_temp       (float   t);

    /// @details Dynamic temperature implementation (a.k.a. entropy) described in the paper https://arxiv.org/abs/2309.02772.
    LLAMA_API struct llama_sampler * llama_sampler_init_temp_ext   (float   t, float   delta, float exponent);

    /// @details XTC sampler as described in https://github.com/oobabooga/text-generation-webui/pull/6335
    LLAMA_API struct llama_sampler * llama_sampler_init_xtc        (float   p, float   t,     size_t min_keep, uint32_t seed);

    /// @details Top n sigma sampling as described in academic paper "Top-nσ: Not All Logits Are You Need" https://arxiv.org/pdf/2411.07641
    LLAMA_API struct llama_sampler * llama_sampler_init_top_n_sigma(float   n);

    /// @details Mirostat 1.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
    /// @param candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
    /// @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
    /// @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
    /// @param m The number of tokens considered in the estimation of `s_hat`. This is an arbitrary value that is used to calculate `s_hat`, which in turn helps to calculate the value of `k`. In the paper, they use `m = 100`, but you can experiment with different values to see how it affects the performance of the algorithm.
    /// @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
    LLAMA_API struct llama_sampler * llama_sampler_init_mirostat(
                             int32_t   n_vocab,
                            uint32_t   seed,
                               float   tau,
                               float   eta,
                             int32_t   m);

    /// @details Mirostat 2.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
    /// @param candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
    /// @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
    /// @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
    /// @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
    LLAMA_API struct llama_sampler * llama_sampler_init_mirostat_v2(
                            uint32_t   seed,
                               float   tau,
                               float   eta);

    /// @details Intializes a GBNF grammar, see grammars/README.md for details.
    /// @param vocab The vocabulary that this grammar will be used with.
    /// @param grammar_str The production rules for the grammar, encoded as a string. Returns an empty grammar if empty. Returns NULL if parsing of grammar_str fails.
    /// @param grammar_root The name of the start symbol for the grammar.
    LLAMA_API struct llama_sampler * llama_sampler_init_grammar(
            const struct llama_vocab * vocab,
                          const char * grammar_str,
                          const char * grammar_root);

    DEPRECATED(LLAMA_API struct llama_sampler * llama_sampler_init_grammar_lazy(
            const struct llama_vocab * vocab,
                          const char * grammar_str,
                          const char * grammar_root,
                         const char ** trigger_words,
                                size_t num_trigger_words,
                   const llama_token * trigger_tokens,
                                size_t num_trigger_tokens),
        "use llama_sampler_init_grammar_lazy_patterns instead");


    /// @details Lazy grammar sampler, introduced in https://github.com/ggml-org/llama.cpp/pull/9639
    /// @param trigger_patterns A list of patterns that will trigger the grammar sampler. Pattern will be matched from the start of the generation output, and grammar sampler will be fed content starting from its first match group.
    /// @param trigger_tokens A list of tokens that will trigger the grammar sampler. Grammar sampler will be fed content starting from the trigger token included.
    LLAMA_API struct llama_sampler * llama_sampler_init_grammar_lazy_patterns(
        const struct llama_vocab * vocab,
                      const char * grammar_str,
                      const char * grammar_root,
                     const char ** trigger_patterns,
                            size_t num_trigger_patterns,
               const llama_token * trigger_tokens,
                            size_t num_trigger_tokens);


    /// NOTE: Avoid using on the full vocabulary as searching for repeated tokens can become slow. For example, apply top-k or top-p sampling first.
    LLAMA_API struct llama_sampler * llama_sampler_init_penalties(
                             int32_t   penalty_last_n,   // last n tokens to penalize (0 = disable penalty, -1 = context size)
                               float   penalty_repeat,   // 1.0 = disabled
                               float   penalty_freq,     // 0.0 = disabled
                               float   penalty_present); // 0.0 = disabled

    ///  @details DRY sampler, designed by p-e-w, as described in: https://github.com/oobabooga/text-generation-webui/pull/5677, porting Koboldcpp implementation authored by pi6am: https://github.com/LostRuins/koboldcpp/pull/982
    LLAMA_API struct llama_sampler * llama_sampler_init_dry(
            const struct llama_vocab *  vocab,
                             int32_t    n_ctx_train,
                               float    dry_multiplier,
                               float    dry_base,
                             int32_t    dry_allowed_length,
                             int32_t    dry_penalty_last_n,
                          const char ** seq_breakers,
                              size_t    num_breakers);

    LLAMA_API struct llama_sampler * llama_sampler_init_logit_bias(
                             int32_t   n_vocab,
                             int32_t   n_logit_bias,
              const llama_logit_bias * logit_bias);

    // this sampler is meant to be used for fill-in-the-middle infilling
    // it's supposed to be used after top_k + top_p sampling
    //
    // 1. if the sum of the EOG probs times the number of candidates is higher than the sum of the other probs -> pick EOG
    // 2. combine probs of tokens that have the same prefix
    //
    // example:
    //
    // - before:
    //   "hel":   0.5
    //   "hell":  0.2
    //   "hello": 0.1
    //   "dummy": 0.1
    //
    // - after:
    //   "hel":   0.8
    //   "dummy": 0.1
    //
    // 3. discard non-EOG tokens with low prob
    // 4. if no tokens are left -> pick EOT
    //
    LLAMA_API struct llama_sampler * llama_sampler_init_infill(const struct llama_vocab * vocab);

    // Returns the seed used by the sampler if applicable, LLAMA_DEFAULT_SEED otherwise
    LLAMA_API uint32_t llama_sampler_get_seed(const struct llama_sampler * smpl);

    /// @details Sample and accept a token from the idx-th output of the last evaluation
    //
    // Shorthand for:
    //    const auto * logits = llama_get_logits_ith(ctx, idx);
    //    llama_token_data_array cur_p = { ... init from logits ... };
    //    llama_sampler_apply(smpl, &cur_p);
    //    auto token = cur_p.data[cur_p.selected].id;
    //    llama_sampler_accept(smpl, token);
    //    return token;
    // Returns the sampled token
    LLAMA_API llama_token llama_sampler_sample(struct llama_sampler * smpl, struct llama_context * ctx, int32_t idx);

    // TODO: extend in the future
    //LLAMA_API void llama_decode_with_sampler(struct llama_context * ctx, struct llama_sampler * smpl, struct llama_batch batch, ...);

    //
    // Model split
    //

    /// @details Build a split GGUF final path for this chunk.
    ///          llama_split_path(split_path, sizeof(split_path), "/models/ggml-model-q4_0", 2, 4) => split_path = "/models/ggml-model-q4_0-00002-of-00004.gguf"
    //  Returns the split_path length.
    LLAMA_API int llama_split_path(char * split_path, size_t maxlen, const char * path_prefix, int split_no, int split_count);

    /// @details Extract the path prefix from the split_path if and only if the split_no and split_count match.
    ///          llama_split_prefix(split_prefix, 64, "/models/ggml-model-q4_0-00002-of-00004.gguf", 2, 4) => split_prefix = "/models/ggml-model-q4_0"
    //  Returns the split_prefix length.
    LLAMA_API int llama_split_prefix(char * split_prefix, size_t maxlen, const char * split_path, int split_no, int split_count);

    // Print system information
    LLAMA_API const char * llama_print_system_info(void);

    // Set callback for all future logging events.
    // If this is not called, or NULL is supplied, everything is output on stderr.
    LLAMA_API void llama_log_set(ggml_log_callback log_callback, void * user_data);

    //
    // Performance utils
    //
    // NOTE: Used by llama.cpp examples, avoid using in third-party apps. Instead, do your own performance measurements.
    //

    struct llama_perf_context_data {
        double t_start_ms;
        double t_load_ms;
        double t_p_eval_ms;
        double t_eval_ms;

        int32_t n_p_eval;
        int32_t n_eval;
    };

    struct llama_perf_sampler_data {
        double t_sample_ms;

        int32_t n_sample;
    };

    LLAMA_API struct llama_perf_context_data llama_perf_context      (const struct llama_context * ctx);
    LLAMA_API void                           llama_perf_context_print(const struct llama_context * ctx);
    LLAMA_API void                           llama_perf_context_reset(      struct llama_context * ctx);

    // NOTE: the following work only with samplers constructed via llama_sampler_chain_init
    LLAMA_API struct llama_perf_sampler_data llama_perf_sampler      (const struct llama_sampler * chain);
    LLAMA_API void                           llama_perf_sampler_print(const struct llama_sampler * chain);
    LLAMA_API void                           llama_perf_sampler_reset(      struct llama_sampler * chain);

    //
    // training
    //

    // function that returns whether or not a given tensor contains trainable parameters
    typedef bool (*llama_opt_param_filter)(const struct ggml_tensor * tensor, void * userdata);

    // always returns true
    LLAMA_API bool llama_opt_param_filter_all(const struct ggml_tensor * tensor, void * userdata);

    struct llama_opt_params {
        uint32_t n_ctx_train; // assumed context size post training, use context size specified in llama_context if 0

        llama_opt_param_filter param_filter; // callback for determining which tensors contain trainable parameters
        void * param_filter_ud;              // userdata for determining which tensors contain trainable parameters

        ggml_opt_get_optimizer_params get_opt_pars; // callback for calculating optimizer parameters
        void * get_opt_pars_ud;                     // userdata for calculating optimizer parameters
    };

    LLAMA_API void llama_opt_init(struct llama_context * lctx, struct llama_model * model, struct llama_opt_params lopt_params);

    LLAMA_API void llama_opt_epoch(
            struct llama_context    * lctx,
            ggml_opt_dataset_t        dataset,
            ggml_opt_result_t         result_train,
            ggml_opt_result_t         result_eval,
            int64_t                   idata_split,
            ggml_opt_epoch_callback   callback_train,
            ggml_opt_epoch_callback   callback_eval);

#ifdef __cplusplus
}
#endif

#endif // LLAMA_H
```