import tensorflow as tf
from classification.model import NoobModel

model = NoobModel()
model.build((1, 256, 256, 3))
model.summary()
