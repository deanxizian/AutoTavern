"""Exact-shape CUDA graphs for deterministic, frozen opponent inference only."""

import torch


class FrozenOpponentGraphs(torch.nn.Module):
    """Capture small batches lazily; larger batches use the unchanged policy.

    Inputs are never padded. Every new graph must exactly match eager logits and
    values. Graphs are process-local caches, not checkpoint state. Callers must
    keep weights fixed and invoke this wrapper in inference mode on one thread.
    """
    def __init__(self, policy, max_batch=16):
        super().__init__()
        if type(max_batch) is not int or not 1 <= max_batch <= 64:
            raise ValueError("Frozen graph batch limit must be an integer in 1..64")
        if policy.training or any(p.requires_grad for p in policy.parameters()):
            raise ValueError("CUDA graphs require an already frozen evaluation policy")
        if getattr(policy, "requires_attachments", False):
            raise ValueError("Variable-length attachment inputs currently require eager inference")
        self.policy = policy
        self.max_batch = max_batch
        self.graphs = {}
        self.graph_calls = self.eager_calls = 0
        self.eval()

    def _capture(self, inputs):
        # Allocate/copy before recording the producer-stream dependency. Waiting
        # before these copies would leave warmup racing their writes.
        static = tuple(x.clone() for x in inputs)
        stream = torch.cuda.Stream(device=inputs[0].device)
        stream.wait_stream(torch.cuda.current_stream(inputs[0].device))
        rng_before = torch.cuda.get_rng_state(inputs[0].device)
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.policy(*static)
        torch.cuda.current_stream(inputs[0].device).wait_stream(stream)
        torch.cuda.synchronize(inputs[0].device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            outputs = self.policy(*static)
        expected = self.policy(*inputs)
        graph.replay()
        if not all(torch.equal(a, b) for a, b in zip(outputs, expected)):
            raise RuntimeError("Frozen CUDA graph differs from eager policy output")
        if not torch.equal(rng_before, torch.cuda.get_rng_state(inputs[0].device)):
            raise RuntimeError("Frozen opponent inference must not consume CUDA RNG")
        self.graphs[len(inputs[0])] = (graph, static, outputs)

    def forward(self, features, cards, masks):
        inputs = (features, cards, masks)
        if self.training or self.policy.training or not torch.is_inference_mode_enabled():
            raise RuntimeError("Frozen opponent graphs are only for evaluation in inference mode")
        if any(x.device != features.device or not x.is_cuda for x in inputs):
            raise ValueError("Frozen graph inputs must be on the same CUDA device")
        if not len(features) or any(len(x) != len(features) for x in inputs):
            raise ValueError("Frozen graph inputs need the same positive batch size")
        if len(features) > self.max_batch:
            self.eager_calls += 1
            return self.policy(*inputs)
        if len(features) not in self.graphs:
            self._capture(inputs)
        graph, static, outputs = self.graphs[len(features)]
        for target, source in zip(static, inputs):
            if (target.shape, target.dtype, target.device) != (source.shape, source.dtype, source.device):
                raise ValueError("Frozen graph input contract changed for a captured batch")
            target.copy_(source)
        graph.replay()
        self.graph_calls += 1
        # Later replay overwrites static outputs. Return owned tensors so a caller
        # can retain predictions across different batches without aliasing them.
        return tuple(x.clone() for x in outputs)

    def inference_stats(self):
        return {"captured_batches": len(self.graphs), "graph_calls": self.graph_calls,
                "eager_calls": self.eager_calls, "max_batch": self.max_batch}
