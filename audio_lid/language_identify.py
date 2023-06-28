import json

import torch
from fairseq.data.text_compressor import TextCompressionLevel, TextCompressor
from fairseq import checkpoint_utils, options, utils
from fairseq.data.audio.raw_audio_dataset import RawAudioDataset, FileAudioDataset
from fairseq.data.data_utils import compute_block_mask_1d

from tqdm import tqdm
import numpy as np


class MemAudioDataset(RawAudioDataset):
    def __init__(
        self,
        samples_list,
        sample_rate,
        max_sample_size=None,
        min_sample_size=0,
        shuffle=True,
        pad=False,
        normalize=False,
        num_buckets=0,
        compute_mask=False,
        text_compression_level=TextCompressionLevel.none,
        **mask_compute_kwargs,
    ):
        super().__init__(
            sample_rate=sample_rate,
            max_sample_size=max_sample_size,
            min_sample_size=min_sample_size,
            shuffle=shuffle,
            pad=pad,
            normalize=normalize,
            compute_mask=compute_mask,
            **mask_compute_kwargs,
        )
        self.samples_list = samples_list

        self.text_compressor = TextCompressor(level=text_compression_level)

        sizes = [16000 for i in range(len(samples_list))]
        self.skipped_indices = set()
        self.sizes = np.array(sizes, dtype=np.int64)
        self.set_bucket_info(num_buckets)

    def __getitem__(self, index):
        wav = self.samples_list[index]

        feats = torch.from_numpy(wav).float()
        feats = self.postprocess(feats, self.sample_rate)

        v = {"id": index, "source": feats}

        if self.is_compute_mask:
            T = self._get_mask_indices_dims(feats.size(-1))
            mask = compute_block_mask_1d(
                shape=(self.clone_batch, T),
                mask_prob=self.mask_prob,
                mask_length=self.mask_length,
                mask_prob_adjust=self.mask_prob_adjust,
                inverse_mask=self.inverse_mask,
                require_same_masks=True,
                expand_adjcent=self.expand_adjacent,
                mask_dropout=self.mask_dropout,
                non_overlapping=self.non_overlapping,
            )

            v["precomputed_mask"] = mask

        return v


class LanguageIdentify:
    def __init__(self, args, top_k=3):
        self.top_k = top_k
        use_cuda = True
        if use_cuda and not torch.cuda.is_available():
            use_cuda = False
        print(f"use_cuda:{use_cuda}")
        # Load model & task
        print("| loading model from {}".format(args.path))
        arg_overrides = {
            "task": {
                "data": args.data
            },
            # 'mask_prob': 0
            # 'max_sample_size': sys.maxsize,
            # 'min_sample_size': 0,
        }
        state = checkpoint_utils.load_checkpoint_to_cpu(args.path, arg_overrides)

        models, _model_args, task = checkpoint_utils.load_model_ensemble_and_task(
            [args.path], arg_overrides=arg_overrides, task=None, state=state
        )
        model = models[0]
        model.eval()
        if use_cuda:
            model.cuda()
        self.model = model
        self.task = task
        self.use_cuda = use_cuda
        self.args = args

    def infer(self, samples_list):
        infer_dataset = MemAudioDataset(
            samples_list,
            sample_rate=self.task.cfg.sample_rate,
            max_sample_size=10 ** 10,  # task.cfg.max_sample_size,
            min_sample_size=1,  # task.cfg.min_sample_size,
            pad=True,
            normalize=self.task.cfg.normalize,
        )

        itr = self.task.get_batch_iterator(
            dataset=infer_dataset,
            max_sentences=1,
            # max_tokens=args.max_tokens,
        ).next_epoch_itr(shuffle=False)

        predictions = {}
        with torch.no_grad():
            for _, sample in tqdm(enumerate(itr)):
                sample = utils.move_to_cuda(sample) if self.use_cuda else sample
                logit = self.model.forward(**sample["net_input"])
                logit_lsm = torch.log_softmax(logit.squeeze(), dim=-1)
                scores, indices = torch.topk(logit_lsm, self.top_k, dim=-1)
                scores = torch.exp(scores).to("cpu").tolist()
                indices = indices.to("cpu").tolist()
                assert sample["id"].numel() == 1
                sample_idx = sample["id"].to("cpu").tolist()[0]
                assert sample_idx not in predictions
                predictions[sample_idx] = [(self.task.target_dictionary[int(i)], s) for s, i in zip(scores, indices)]

        return predictions


if __name__ == "__main__":
    np.random.seed(123)
    # Parse command-line arguments for generation
    parser = options.get_generation_parser(default_task="audio_classification")
    # parser.add_argument('--infer-merge', type=str, default='mean')
    parser.add_argument("--infer-xtimes", type=int, default=1)
    parser.add_argument("--infer-num-samples", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--infer-max-sample-size", type=int, default=5 * 16000
    )  # 5 secs
    parser.add_argument("--infer-manifest", required="./pretrain", type=str)
    parser.add_argument("--output-dir", default="./temp", type=str)

    args = options.parse_args_and_arch(parser)
    # Setup task
    # task = tasks.setup_task(args)
    use_cuda = not args.cpu
    if use_cuda and not torch.cuda.is_available():
        use_cuda = False
    print(f"use_cuda:{use_cuda}")
    # Load model & task
    print("| loading model from {}".format(args.path))
    arg_overrides = {
        "task": {
            "data": args.data
        },
        # 'mask_prob': 0
        # 'max_sample_size': sys.maxsize,
        # 'min_sample_size': 0,
    }
    state = checkpoint_utils.load_checkpoint_to_cpu(args.path, arg_overrides)

    models, _model_args, task = checkpoint_utils.load_model_ensemble_and_task(
        [args.path], arg_overrides=arg_overrides, task=None, state=state
    )
    model = models[0]
    model.eval()
    if use_cuda:
        model.cuda()
    # Load dataset

    infer_manifest = args.infer_manifest
    infer_dataset = FileAudioDataset(
        infer_manifest,
        sample_rate=task.cfg.sample_rate,
        max_sample_size=10 ** 10,  # task.cfg.max_sample_size,
        min_sample_size=1,  # task.cfg.min_sample_size,
        pad=True,
        normalize=task.cfg.normalize,
    )

    itr = task.get_batch_iterator(
        dataset=infer_dataset,
        max_sentences=1,
        # max_tokens=args.max_tokens,
        num_workers=4,
    ).next_epoch_itr(shuffle=False)
    predictions = {}

    with torch.no_grad():
        for _, sample in tqdm(enumerate(itr)):
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            logit = model.forward(**sample["net_input"])
            logit_lsm = torch.log_softmax(logit.squeeze(), dim=-1)
            scores, indices = torch.topk(logit_lsm, args.top_k, dim=-1)
            scores = torch.exp(scores).to("cpu").tolist()
            indices = indices.to("cpu").tolist()
            assert sample["id"].numel() == 1
            sample_idx = sample["id"].to("cpu").tolist()[0]
            assert sample_idx not in predictions
            predictions[sample_idx] = [(task.target_dictionary[int(i)], s) for s, i in zip(scores, indices)]

    with open(f"{args.output_dir}/predictions.txt", "w") as fo:
        for idx in range(len(infer_dataset)):
            fo.write(json.dumps(predictions[idx]) + "\n")

    print(f"Outputs will be located at - {args.output_dir}/predictions.txt")
