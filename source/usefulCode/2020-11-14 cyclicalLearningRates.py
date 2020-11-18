## Cyclical Learning Rates

# [scale_fn error](https://github.com/tensorflow/addons/issues/1158)


from tensorflow_addons.optimizers import cyclical_learning_rate

clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))

CLR_CALLBACK = cyclical_learning_rate.CyclicalLearningRate(initial_learning_rate = 1e-7
                                                    ,maximal_learning_rate = 1e-4
                                                    ,scale_fn=clr_fn
                                                    ,step_size=4)


# callback = cyclical_learning_rate.ExponentialCyclicalLearningRate(
#     initial_learning_rate=1e-8,
#     maximal_learning_rate=1e-2,
#     step_size=100,
# )

model_config.update(EXPERIMENT = 'FC 32 Dropout 16 8 CLR')
model_config.update(DROPOUT = 0.3)
model_config.update(EPOCHS = 500)

