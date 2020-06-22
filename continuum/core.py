from continuum.models.gauss.deep import BaseDeepGPSystem

print(BaseDeepGPSystem)
# from continuum.data.loaders import ElevatorLoader


# # # this is for running the notebook in our testing framework
# smoke_test = ('CI' in os.environ)

# elevator_loader = ElevatorLoader()
# train_loader, test_loader, x_shape = elevator_loader.get_data()


# num_output_dims = 2 if smoke_test else 10


# class DeepGPSystem(DeepGP):
#     def __init__(self, train_x_shape):
#         hidden_layer = DeepGPHiddenLayer(
#             input_dims=train_x_shape[-1],
#             output_dims=num_output_dims,
#             mean_type='linear',
#         )

#         last_layer = DeepGPHiddenLayer(
#             input_dims=hidden_layer.output_dims,
#             output_dims=None,
#             mean_type='constant',
#         )

#         super().__init__()

#         self.hidden_layer = hidden_layer
#         self.last_layer = last_layer
#         self.likelihood = GaussianLikelihood()

#     def forward(self, inputs):
#         hidden_rep1 = self.hidden_layer(inputs)
#         output = self.last_layer(hidden_rep1)
#         return output

#     def predict(self, test_loader):
#         with torch.no_grad():
#             mus = []
#             variances = []
#             lls = []
#             for x_batch, y_batch in test_loader:
#                 preds = model.likelihood(model(x_batch))
#                 mus.append(preds.mean)
#                 variances.append(preds.variance)
#                 lls.append(model.likelihood.log_marginal(y_batch, model(x_batch)))

#         return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1), torch.cat(lls, dim=-1)




# model = DeepGPSystem(x_shape)
# if torch.cuda.is_available():
#     model = model.cuda()

# num_epochs = 1 if smoke_test else 10
# num_samples = 3 if smoke_test else 10


# optimizer = torch.optim.Adam([
#     {'params': model.parameters()},
# ], lr=0.01)
# mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model, x_shape[-2]))

# epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")
# for i in epochs_iter:
#     # Within each iteration, we will go over each minibatch of data
#     minibatch_iter = tqdm.tqdm(train_loader, desc="Minibatch", leave=False)
#     for x_batch, y_batch in minibatch_iter:
#         with gpytorch.settings.num_likelihood_samples(num_samples):
#             optimizer.zero_grad()
#             output = model(x_batch)
#             loss = -mll(output, y_batch)
#             loss.backward()
#             optimizer.step()

#             minibatch_iter.set_postfix(loss=loss.item())


# # test_dataset = TensorDataset(test_x, test_y)
# # test_loader = DataLoader(test_dataset, batch_size=1024)

# model.eval()
# predictive_means, predictive_variances, test_lls = model.predict(test_loader)

# logger.info((predictive_means, predictive_variances, predictive_variances))

# # rmse = torch.mean(torch.pow(predictive_means.mean(0) - test_y, 2)).sqrt()
# # logger.warning(f"RMSE: {rmse.item()}, NLL: {-test_lls.mean().item()}")