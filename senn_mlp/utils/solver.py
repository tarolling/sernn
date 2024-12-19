from functools import partial

import jax
import jax.numpy as jnp
import flax


class Solver:
    """probably should be further factored"""

    def __init__(self, cfg, template, task, key, example):
        self.cfg = cfg["opt"]
        self.template = template
        self.task = task
        self.model = self.template.build()
        self.state = self.model.init(key, example)
        print(f"contents: {self.template.contents}")

        def nonincreasing(seq):
            return all(x >= y for x, y in zip(seq, seq[1:]))

        assert nonincreasing(
            [c for c in self.template.contents if c is not None]
        ), "model.restrict_params is broken for increasing feature sizes"
        self.state = self.model.apply(
            self.state, self.template.contents, method=self.model.restrict_params
        )
        self.optimizer, self.opt_state = self.make_opt(self.state)
        self.lr = cfg["opt"]["lr"].get()
        self.recompile()
        self.last_natlen = None
        self.weight_decay = cfg["opt"]["weight_decay"].get(0.0)

    def item_loss(self, state, datum, label):
        y = self.model.apply(state, datum)
        loss = self.task.loss_function(label, y)
        return loss

    def item_correct(self, state, datum, label):
        y = self.model.apply(state, datum)
        return jnp.argmax(y) == label

    def _batch_loss(self, state, data, labels):
        """NOTE: This will not account for a changing model -
        if model changes this must be re-jitted"""
        return jax.vmap(partial(self.item_loss, state))(data, labels).mean()

    def _batch_acc(self, state, data, labels):
        """NOTE: This will not account for a changing model -
        if model changes this must be re-jitted"""
        return jax.vmap(partial(self.item_correct, state))(data, labels).mean()

    def make_opt(self, state):
        opt = CGNG(self.cfg, state["params"])
        opt_state = opt.init(flax.core.frozen_dict.freeze({}))
        return opt, opt_state

    def restrict_grad(self, grad):
        variables = {"params": grad}
        return self.model.apply(
            variables, self.template.contents, method=self.model.restrict_grad
        )["params"]

    def recompile(self):
        # self.restrict_grad = lambda grad: self.model.apply(
        #     {'params': grad}, self.template.contents, method=self.model.restrict_grad)['params']
        func = lambda params, x: self.model.apply(
            self.state.copy({"params": params}), x
        )
        self.observe = jax.jit(
            partial(
                self.optimizer.observe,
                func,
                self.task.loss_function,
                self.restrict_grad,
            )
        )
        self.apply_model = jax.jit(self.model.apply)
        self.batch_loss = jax.jit(self._batch_loss)
        self.batch_acc = jax.jit(self._batch_acc)

    def update_params(self, mul, tan, weight_decay=0.0):
        # apply weight decay to tangent
        tan = jtm(
            lambda t, s: 1 / (1 + weight_decay) * t
            + weight_decay / (1 + weight_decay) * s,
            tan,
            self.state["params"],
        )
        tan = self.restrict_grad(tan)
        # apply tangent to state with learning rate
        new = jtm(lambda t, s: s + -mul * self.lr * t, tan, self.state["params"])
        self.state = self.state.copy({"params": new})

    def train_batch(self, batch, observe_only=False, loud=False):
        data, labels = batch
        loss = self.batch_loss(self.state, data, labels)
        if loud:
            print(f"loss: {loss:.3E}")
        summary.scalar("loss", loss)
        summary.scalar(
            "features", sum([c for c in self.template.contents if c is not None])
        )
        for i, f in enumerate(self.template.contents):
            summary.scalar(f"features_{i}", f if f is not None else 0)
        self.opt_state = self.observe(
            data, labels, self.state["params"], self.opt_state
        )

        grad = self.optimizer.SG.read(self.opt_state)
        ngrad = self.optimizer.read(self.opt_state)
        nat_len = tree_dot(ngrad, grad)
        self.last_natlen = nat_len
        summary.scalar("baseline", nat_len)
        summary.scalar("normed_baseline", nat_len / loss)

        summary.scalar("param_Fnorm", self.optimizer.param_Fnorm.read(self.opt_state))
        summary.scalar(
            "param_l2norm", tree_dot(self.state["params"], self.state["params"])
        )

        # summary.scalar("fresh_baseline", self.eval_feature_proposal(batch, self.state))

        if not observe_only:
            # self.update_params(loss/nat_len, ngrad)
            self.update_params(1.0, ngrad, weight_decay=self.weight_decay)
        if loud:
            return loss

    def eval_feature_proposal(self, batch, fstate):
        assert self.cfg["tau"].get() is None
        assert (
            False
        ), "nat_len calculation is inconsistent with that of the training update"
        data, labels = batch
        _, fopt_state = self.make_opt(fstate)
        # fopt_state = self.observe(data, labels, fstate['params'], fopt_state)
        fopt_state = self.observe(data, labels, fstate["params"], self.opt_state)

        grad = self.optimizer.SG.read(fopt_state)
        ngrad = self.optimizer.read(fopt_state)
        nat_len = tree_dot(ngrad, grad)
        return nat_len

    def test_batch(self, batch):
        # print("test batch...")
        data, labels = batch
        loss = self.batch_loss(self.state, data, labels)
        # print(f"validation loss: {loss:.3E}")
        summary.scalar("validation loss", loss)

    def test_acc(self, batch):
        data, labels = batch
        acc = self.batch_acc(self.state, data, labels)
        summary.scalar("validation accuracy", acc)

    def train_acc(self, batch):
        data, labels = batch
        acc = self.batch_acc(self.state, data, labels)
        summary.scalar("training accuracy", acc)
