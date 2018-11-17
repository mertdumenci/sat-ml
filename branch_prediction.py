import pickle

from satml.expression import Type, _Expression, expr, pprint
from satml.solver     import simplify

import tensorflow.keras
from   tensorflow.keras.preprocessing.sequence import pad_sequences
from   tensorflow.keras.utils                  import to_categorical
from   tensorflow.keras.models                 import Sequential
from   tensorflow.keras.layers                 import Dense, Embedding, LSTM

import numpy as np
from sklearn.model_selection import train_test_split


def rename_vars(an_exp):
    """
    Normalizes/renames all variables in a formula.
    Returns `renamed exp, mapping, num free variables`.
    """
    r_exp, mapping, current = _rename_vars(an_exp, dict(), 1)
    return r_exp, mapping, (current - 1)


def _rename_vars(an_exp, mapping, current):
    if an_exp is None:
        return an_exp, mapping, current
    elif an_exp.typ == Type.VAR:
        val = an_exp.l_val

        if val in mapping:
            val = mapping[val]
        else:
            mapping[val] = current
            val = current
            current = current + 1
        return expr((Type.VAR, val, None)), mapping, current
    elif an_exp.typ == Type.CONST:
        return an_exp, mapping, current
    
    l_exp, mapping, current = _rename_vars(an_exp.l_val, mapping, current)
    r_exp, mapping, current = _rename_vars(an_exp.r_val, mapping, current)
    
    return expr((an_exp.typ, l_exp, r_exp)), mapping, current


_STOP_WORD = 0
_CONNECTIVE_ENCODING = {
    Type.AND: -1,
    Type.OR:  -2,
}


def _encode_expression(an_exp):
    """
    Expects simplified, normalized formulas.
    See `satml.solver.simplify` and `rename_vars`.
    """
    if not isinstance(an_exp, _Expression):
        return an_exp
    
    if an_exp.typ == Type.VAR:
        var = an_exp.l_val
        return [var]
    if an_exp.typ == Type.NOT:
        # Inner has to be a variable by the postconditions of `simplify`.
        var = -an_exp.l_val.l_val
        return [var]
    
    return ([_CONNECTIVE_ENCODING[an_exp.typ]]
            + [_STOP_WORD]
            + _encode_expression(an_exp.l_val)
            + [_STOP_WORD]
            + _encode_expression(an_exp.r_val))


def encode_expression(an_exp):
    s_exp, mapping, _ = rename_vars(simplify(an_exp))
    encoded = _encode_expression(s_exp)
    return encoded, mapping


print('Loading data...')
with open('all_history.pickle', 'rb') as f:
    all_history = pickle.load(f)

print('Loaded {} formulas. Preparing data...'.format(len(all_history)))
sequences, labels = [], []
for expression, _, best_branch, _ in all_history:
    # We don't really care about expressions with <= 1 variables.
    if expression.typ != Type.AND and expression.typ != Type.OR:
        continue
        
    encoded, mapping = encode_expression(expression)
    mapped_best_branch = mapping[best_branch]

    sequences.append(encoded)
    labels.append(mapped_best_branch)

X = pad_sequences(sequences)
y = to_categorical(np.array(labels))

num_features  = np.max(X) - np.min(X) + 1
_, num_labels = y.shape
batch_size    = 32

# `keras.layers.Embedding` doesn't like negative values.
X = X - np.min(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print('Building model...')
model = Sequential([
    Embedding(num_features, 128),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True),
    LSTM(80, dropout=0.2, recurrent_dropout=0.2, return_sequences=True),
    LSTM(40, dropout=0.2, recurrent_dropout=0.2),
    Dense(num_labels, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=6,
          validation_data=(X_test, y_test))
score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)

print('Test score:', score)
print('Test accuracy:', acc)
