import torch
from transformer_modeling import *
from transformer_training import *


def demo_simple_copy_task():
    print('Train the simple copy task.')
    V = 11
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    model = make_model(V, V, N=2)
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    for epoch in range(10):
        model.train()
        run_epoch(data_gen(V, 30, 20), model,
                  SimpleLossCompute(model.generator, criterion, model_opt))
        model.eval()
        print(run_epoch(data_gen(V, 30, 5), model,
                        SimpleLossCompute(model.generator, criterion, None)))

    print('Inference: copy task')
    model.eval()
    src_var = Variable(torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]))
    src_mask = Variable(torch.ones(1, 1, 10))
    print(greedy_decode(model, src_var, src_mask, max_len=10, start_symbol=1))


def demo_small_model():
    # Small example model.
    tmp_model = make_model(10, 10, 2)
    print(tmp_model)


if __name__ == '__main__':
    demo_simple_copy_task()
