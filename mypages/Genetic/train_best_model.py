#from mypages.Genetic.model_operation import My_model, Create_model, Dataset
import pickle
import torch



with open(f'result_model10.pickle', 'rb') as f:
    model = pickle.load(f)
    torch.save(model.model.state_dict(), "model.pth")
with open(f'Scaler.pickle', 'wb') as f:
    pickle.dump(model.dataset.Scaler, f)


# vals = [0.09285523935399029, 0.1003677320028289, 0.11742471059302935, 0.11982630103659093, 0.12133463751315401]
# bots = [[12, 0, 0, 0, 0, 1, 0, 3, 0, 3, 0, 0, 1, 0, 12, 0, 0, 0, 0],
#  [12, 0, 0, 0, 0, 1, 0, 3, 1, 3, 0, 0, 1, 0, 12, 1, 0, 0, 0],
#  [12, 0, 0, 0, 0, 1, 0, 3, 1, 3, 0, 0, 1, 0, 12, 0, 0, 0, 0],
#  [10, 0, 0, 0, 0, 1, 0, 3, 1, 1, 0, 0, 1, 0, 12, 0, 0, 0, 0],
#  [12, 0, 0, 0, 0, 1, 0, 3, 0, 3, 0, 0, 1, 0, 11, 0, 0, 0, 0]]
#
# path = '/home/dmitry/PycharmProjects/Creadit_card/dataset/creditcard_2023.csv'
# test = 0.1
# batch = 128
# dataset = Dataset(path, header=0, norm=2, test=test, batch=batch)
#
# for i in range(0,5):
#     model = Create_model(bots[i], 29)
#     example_model = My_model(model, dataset)
#     lrs = [0.001, 0.0007, 0.0004, 0.0001, 0.00007, 0.00003, 0.00001]
#     example_model.train(epochs=20, lrs=lrs)
#     example_model.show_test_val()
#     with open(f'result_model{i}0.pickle', 'wb') as f:
#         pickle.dump(example_model, f)

history = []

for i in range(3):
    with open(f'result_model{i}0.pickle', 'rb') as f:
        model = pickle.load(f)
        history.append(model.history)

with open(f'history_models.pickle', 'wb') as f:
    pickle.dump(history, f)

