# ML Assignment 2 - Bagging, Boosting and Bias-Variance on Olivetti Faces
# Run with:  python assignment2_simple.py
# Everything (log, plots, split) is saved in the folder "outputs"

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")                      # so plots can be saved without a window
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import fetch_olivetti_faces

# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------
ROLL_NO = "M26MAC012"
NUM_CLASSES = 40
N_REPEATS = 30                 # number of resampled training sets for Q3
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

# seed = all the digits of the roll number, in order
seed_text = ""
for ch in ROLL_NO:
    if ch.isdigit():
        seed_text = seed_text + ch
SEED = int(seed_text)                      # 26012

logfile = open(OUT + "/log.txt", "w")


def log(*args):
    """print and also save to the log file"""
    text = " ".join([str(a) for a in args])
    print(text)
    logfile.write(text + "\n")
    logfile.flush()


# ------------------------------------------------------------------
# Small helper functions
# ------------------------------------------------------------------
def accuracy(pred, y):
    return float(np.mean(pred == y))


def count_votes(all_preds):
    """all_preds has shape (number of models, number of samples).
    Returns an array (samples, classes) with the number of votes for each class."""
    n_samples = all_preds.shape[1]
    votes = np.zeros((n_samples, NUM_CLASSES))
    for model_pred in all_preds:
        for i in range(n_samples):
            votes[i, model_pred[i]] += 1
    return votes


def pairwise_disagreement(all_preds):
    """average (over all pairs of trees and all samples) of how often two trees differ"""
    n_models = all_preds.shape[0]
    if n_models < 2:
        return float("nan")
    votes = count_votes(all_preds)
    # for each sample count the pairs of trees that agree
    agree_pairs = (votes * (votes - 1)).sum(axis=1) / (n_models * (n_models - 1))
    return float(np.mean(1 - agree_pairs))


def vote_margin(votes):
    """winning vote fraction minus runner up vote fraction, averaged over samples"""
    fractions = votes / votes.sum(axis=1, keepdims=True)
    fractions = np.sort(fractions, axis=1)
    return float(np.mean(fractions[:, -1] - fractions[:, -2]))


def leaf_info(tree, n_train):
    """number of leaves, mean samples per leaf, fraction of leaves with 1 example"""
    t = tree.tree_
    is_leaf = t.children_left == -1
    n_leaves = int(is_leaf.sum())
    single = float(np.mean(t.n_node_samples[is_leaf] == 1))
    return n_leaves, n_train / n_leaves, single


def one_hot(pred):
    v = np.zeros((len(pred), NUM_CLASSES))
    for i in range(len(pred)):
        v[i, pred[i]] = 1
    return v


def show_face(ax, x, title):
    ax.imshow(x.reshape(64, 64), cmap="gray")
    ax.set_title(title, fontsize=7)
    ax.axis("off")


# ------------------------------------------------------------------
# Bagging (written from scratch)
# ------------------------------------------------------------------
def train_bagging(X, y, n_trees, depth, min_leaf, rng):
    """Train n_trees trees, each on a bootstrap sample of the same size as X.
    Returns the trees, a table telling which samples were inside each bootstrap
    sample, and the number of distinct samples in each bootstrap sample."""
    n = len(X)
    trees = []
    in_bag = np.zeros((n_trees, n), dtype=bool)
    distinct_counts = []
    for b in range(n_trees):
        idx = rng.integers(0, n, n)                  # sample with replacement
        tree = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=min_leaf,
                                      random_state=int(rng.integers(0, 1000000)))
        tree.fit(X[idx], y[idx])
        trees.append(tree)
        in_bag[b, np.unique(idx)] = True
        distinct_counts.append(len(np.unique(idx)))
    return trees, in_bag, distinct_counts


def trees_predict(trees, X):
    """predictions of every tree, shape (number of trees, number of samples)"""
    return np.array([t.predict(X) for t in trees])


# ------------------------------------------------------------------
# SAMME AdaBoost (written from scratch)
# ------------------------------------------------------------------
def train_samme(X, y, depth, rounds, K):
    """Multiclass AdaBoost (SAMME). Returns a dictionary with everything we need."""
    n = len(X)
    w = np.ones(n) / n                          # start with equal weights
    result = {"learners": [], "alphas": [], "errors": [],
              "w_before": [], "w_after": [], "preds": [], "stopped": None}

    for t in range(rounds):
        learner = DecisionTreeClassifier(max_depth=depth, random_state=0)
        learner.fit(X, y, sample_weight=w)
        pred = learner.predict(X)
        wrong = (pred != y)
        error = float(np.sum(w[wrong]))         # weighted error

        # weak learner not better than random guessing among K classes
        if error >= 1 - 1.0 / K:
            result["stopped"] = "round " + str(t + 1) + ": error " + str(error) + \
                                " is not better than random, learner discarded and boosting stopped"
            break

        error_safe = max(error, 1e-12)          # avoid dividing by zero if error is 0
        alpha = np.log((1 - error_safe) / error_safe) + np.log(K - 1)

        new_w = w * np.exp(alpha * wrong)       # increase weight of wrong samples
        new_w = new_w / new_w.sum()             # normalise

        result["learners"].append(learner)
        result["alphas"].append(alpha)
        result["errors"].append(error)
        result["w_before"].append(w.copy())
        result["w_after"].append(new_w.copy())
        result["preds"].append(pred)
        w = new_w

        if error == 0:
            result["stopped"] = "round " + str(t + 1) + ": training error is 0, stopped"
            break

    result["alphas"] = np.array(result["alphas"])
    result["errors"] = np.array(result["errors"])
    result["w_before"] = np.array(result["w_before"])
    result["w_after"] = np.array(result["w_after"])
    result["preds"] = np.array(result["preds"])
    return result


def samme_learner_preds(model, X):
    return np.array([l.predict(X) for l in model["learners"]])


def samme_scores(learner_preds, alphas, rounds, K):
    """S_k(x) = sum of alpha_t for learners (first 'rounds' of them) that vote class k.
    q_k(x) = S_k(x) / sum of alpha_t"""
    rounds = min(rounds, len(alphas))
    S = np.zeros((learner_preds.shape[1], K))
    for t in range(rounds):
        for i in range(learner_preds.shape[1]):
            S[i, learner_preds[t][i]] += alphas[t]
    q = S / np.sum(alphas[:rounds])
    return S, q


# ------------------------------------------------------------------
# Load data and make the split
# ------------------------------------------------------------------
data = fetch_olivetti_faces()
X_all = data.data                    # (400, 4096) flattened pixels
y_all = data.target                  # (400,)

rng = np.random.default_rng(SEED)
train_idx, val_idx, test_idx = [], [], []
for c in range(NUM_CLASSES):
    idx_c = np.where(y_all == c)[0]               # the 10 images of person c
    idx_c = rng.permutation(idx_c)
    train_idx.extend(idx_c[0:6])
    val_idx.extend(idx_c[6:8])
    test_idx.extend(idx_c[8:10])
train_idx, val_idx, test_idx = np.array(train_idx), np.array(val_idx), np.array(test_idx)
np.savez(OUT + "/split.npz", train=train_idx, val=val_idx, test=test_idx, seed=SEED)

X_train, y_train = X_all[train_idx], y_all[train_idx]
X_val, y_val = X_all[val_idx], y_all[val_idx]
X_test, y_test = X_all[test_idx], y_all[test_idx]

# masked test set: set the central 16x16 block (rows and cols 24..39) to zero
X_test_masked = X_test.copy().reshape(-1, 64, 64)
X_test_masked[:, 24:40, 24:40] = 0
X_test_masked = X_test_masked.reshape(len(X_test), -1)

# pixel numbers that belong to the masked block
mask_pixels = []
for r in range(24, 40):
    for c in range(24, 40):
        mask_pixels.append(r * 64 + c)

n_train = len(X_train)
log("seed =", SEED, "| train/val/test sizes =", len(X_train), len(X_val), len(X_test))

# ==================================================================
# Q1 (a): single decision tree baseline
# ==================================================================
log("\n===== Q1(a) single tree grid =====")
best_val = -1
best_depth = None
best_leaf = None
for depth in [2, 4, 8, 16, None]:
    for min_leaf in [1, 2, 4]:
        tree = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=min_leaf, random_state=0)
        tree.fit(X_train, y_train)
        tr_acc = accuracy(tree.predict(X_train), y_train)
        va_acc = accuracy(tree.predict(X_val), y_val)
        te_acc = accuracy(tree.predict(X_test), y_test)
        n_leaves, per_leaf, single = leaf_info(tree, n_train)
        log("depth", depth, "min_leaf", min_leaf, "| train %.3f val %.3f test %.3f" % (tr_acc, va_acc, te_acc),
            "| leaves", n_leaves, "samples/leaf %.2f" % per_leaf, "single-example leaves %.2f" % single)
        # choose by validation accuracy only (on a tie keep the first = simpler setting)
        if va_acc > best_val:
            best_val, best_depth, best_leaf = va_acc, depth, min_leaf

log("SELECTED single tree: max_depth =", best_depth, ", min_samples_leaf =", best_leaf,
    ", val acc = %.3f" % best_val)

tree = DecisionTreeClassifier(max_depth=best_depth, min_samples_leaf=best_leaf, random_state=0)
tree.fit(X_train, y_train)

# ==================================================================
# Q1 (b) and (c): bagging
# ==================================================================
log("\n===== Q1(b),(c) bagging =====")
B_LIST = [1, 5, 10, 25, 50, 100, 200]
# train 200 trees once. Bagging with B trees = the first B of them.
bag_rng = np.random.default_rng(SEED + 1)
bag_trees, in_bag, distinct = train_bagging(X_train, y_train, 200, best_depth, best_leaf, bag_rng)

preds_train = trees_predict(bag_trees, X_train)
preds_val = trees_predict(bag_trees, X_val)
preds_test = trees_predict(bag_trees, X_test)
preds_test_masked = trees_predict(bag_trees, X_test_masked)

expected_distinct = n_train * (1 - (1 - 1.0 / n_train) ** n_train)
log("theoretical expected distinct examples per bootstrap sample = %.2f" % expected_distinct)

oob_list, dis_list, margin_list, val_list = [], [], [], []
for B in B_LIST:
    votes_tr = count_votes(preds_train[:B])
    votes_va = count_votes(preds_val[:B])
    votes_te = count_votes(preds_test[:B])

    # out of bag votes: for every training sample only use trees that did NOT see it
    oob_votes = np.zeros((n_train, NUM_CLASSES))
    for b in range(B):
        for i in range(n_train):
            if not in_bag[b, i]:
                oob_votes[i, preds_train[b][i]] += 1
    has_oob = oob_votes.sum(axis=1) > 0
    oob_acc = accuracy(oob_votes[has_oob].argmax(axis=1), y_train[has_oob])

    dis = pairwise_disagreement(preds_val[:B])          # measured on validation set
    marg = vote_margin(votes_va)                        # measured on validation set
    val_acc = accuracy(votes_va.argmax(axis=1), y_val)

    oob_list.append(oob_acc)
    dis_list.append(dis)
    margin_list.append(marg)
    val_list.append(val_acc)
    log("B=%3d | train %.3f val %.3f test %.3f | distinct %.1f | OOB %.3f (covers %.2f of train) | disagreement %.3f | margin %.3f"
        % (B, accuracy(votes_tr.argmax(axis=1), y_train), val_acc, accuracy(votes_te.argmax(axis=1), y_test),
           np.mean(distinct[:B]), oob_acc, has_oob.mean(), dis, marg))

# choose B with best validation accuracy (smallest B on a tie)
best_i = 0
for i in range(len(B_LIST)):
    if val_list[i] > val_list[best_i]:
        best_i = i
B_FINAL = B_LIST[best_i]
log("SELECTED B_final =", B_FINAL)

plt.figure(figsize=(11, 3))
for k, (values, name) in enumerate([(oob_list, "OOB accuracy"),
                                    (dis_list, "pairwise disagreement (val)"),
                                    (margin_list, "mean vote margin (val)")]):
    plt.subplot(1, 3, k + 1)
    plt.plot(B_LIST, values, "o-")
    plt.xscale("log")
    plt.xlabel("B")
    plt.title(name)
plt.tight_layout()
plt.savefig(OUT + "/q1c_plots.png", dpi=200)
plt.close()

# ==================================================================
# Q1 (d): single tree vs bagging, clean and masked
# ==================================================================
log("\n===== Q1(d) tree vs bagging =====")
votes_clean = count_votes(preds_test[:B_FINAL])
votes_masked = count_votes(preds_test_masked[:B_FINAL])
tree_proba_clean = tree.predict_proba(X_test)
tree_proba_masked = tree.predict_proba(X_test_masked)
tree_pred_clean = tree.classes_[tree_proba_clean.argmax(axis=1)]
tree_pred_masked = tree.classes_[tree_proba_masked.argmax(axis=1)]

log("clean : tree acc %.3f conf %.3f | bagging acc %.3f conf %.3f" %
    (accuracy(tree_pred_clean, y_test), tree_proba_clean.max(axis=1).mean(),
     accuracy(votes_clean.argmax(axis=1), y_test), (votes_clean.max(axis=1) / B_FINAL).mean()))
log("masked: tree acc %.3f conf %.3f | bagging acc %.3f conf %.3f" %
    (accuracy(tree_pred_masked, y_test), tree_proba_masked.max(axis=1).mean(),
     accuracy(votes_masked.argmax(axis=1), y_test), (votes_masked.max(axis=1) / B_FINAL).mean()))

bag_pred_clean = votes_clean.argmax(axis=1)
cases = []          # (description, test index)
fixed = [i for i in range(len(y_test)) if bag_pred_clean[i] == y_test[i] and tree_pred_clean[i] != y_test[i]]
both_wrong = [i for i in range(len(y_test)) if bag_pred_clean[i] != y_test[i] and tree_pred_clean[i] != y_test[i]]
for i in fixed[:2]:
    cases.append(("bagging corrects tree", i))
for i in both_wrong[:2]:
    cases.append(("both wrong", i))

plt.figure(figsize=(8, 2.3 * max(len(cases), 1)))
for r, (name, i) in enumerate(cases):
    top5 = np.argsort(-votes_clean[i])[:5]
    log(name, "| test face", i, "| true", y_test[i], "| tree predicted", tree_pred_clean[i],
        "| bagging predicted", bag_pred_clean[i], "| top votes (class:votes):",
        [(int(c), int(votes_clean[i, c])) for c in top5], "| votes for true class:", int(votes_clean[i, y_test[i]]))
    plt.subplot(len(cases), 2, 2 * r + 1)
    plt.imshow(X_test[i].reshape(64, 64), cmap="gray")
    plt.title("%s\ntrue=%d tree=%d bag=%d" % (name, y_test[i], tree_pred_clean[i], bag_pred_clean[i]), fontsize=7)
    plt.axis("off")
    plt.subplot(len(cases), 2, 2 * r + 2)
    colors = ["green" if c == y_test[i] else "blue" for c in range(NUM_CLASSES)]
    plt.bar(range(NUM_CLASSES), votes_clean[i], color=colors)
    plt.title("bagging votes (green = true class)", fontsize=7)
plt.tight_layout()
plt.savefig(OUT + "/q1d_cases.png", dpi=200)
plt.close()

# ==================================================================
# Q2 (a): SAMME on a tiny 3 class example, log one complete round
# ==================================================================
log("\n===== Q2(a) toy SAMME example (K=3, 9 samples) =====")
X_toy = np.array([[0.2, 0.3], [0.8, 0.9], [1.6, 0.1], [1.4, 0.7], [2.6, 0.4],
                  [3.2, 0.8], [3.0, 0.2], [4.1, 0.6], [4.8, 0.5]])
y_toy = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
toy = train_samme(X_toy, y_toy, depth=1, rounds=3, K=3)
log("initial weights    :", np.round(toy["w_before"][0], 4))
log("true labels        :", y_toy)
log("weak learner preds :", toy["preds"][0])
log("weighted error     : %.4f   (random guessing level for K=3 is %.4f)" % (toy["errors"][0], 1 - 1.0 / 3))
log("alpha              : %.4f" % toy["alphas"][0])
log("updated weights    :", np.round(toy["w_after"][0], 4))

# ==================================================================
# Q2 (b): choose weak learner depth and number of rounds
# ==================================================================
log("\n===== Q2(b) boosting grid =====")
ROUND_LIST = [10, 25, 50, 100]
best_val = -1
BOOST_DEPTH, BOOST_ROUNDS = None, None
for depth in [1, 2, 3, 4]:
    model = train_samme(X_train, y_train, depth, 100, NUM_CLASSES)
    p_tr = samme_learner_preds(model, X_train)
    p_va = samme_learner_preds(model, X_val)
    for rounds in ROUND_LIST:
        tr_acc = accuracy(samme_scores(p_tr, model["alphas"], rounds, NUM_CLASSES)[0].argmax(axis=1), y_train)
        va_acc = accuracy(samme_scores(p_va, model["alphas"], rounds, NUM_CLASSES)[0].argmax(axis=1), y_val)
        log("depth", depth, "rounds", rounds, "| train %.3f val %.3f" % (tr_acc, va_acc))
        if va_acc > best_val:                          # strict > keeps the smaller setting on ties
            best_val, BOOST_DEPTH, BOOST_ROUNDS = va_acc, depth, rounds
log("SELECTED boosting: depth =", BOOST_DEPTH, ", rounds =", BOOST_ROUNDS, ", val acc = %.3f" % best_val)

boost = train_samme(X_train, y_train, BOOST_DEPTH, BOOST_ROUNDS, NUM_CLASSES)
if boost["stopped"] is not None:
    log("note:", boost["stopped"])
n_rounds_used = len(boost["alphas"])

# weight of the 10 heaviest training images in every round
top10_weight = []
for t in range(n_rounds_used):
    top10_weight.append(np.sort(boost["w_before"][t])[-10:].sum())

plt.figure(figsize=(14, 3))
plot_items = [(boost["errors"], "weighted training error"),
              (boost["alphas"], "learner weight alpha"),
              (boost["w_before"].max(axis=1), "maximum sample weight"),
              (top10_weight, "total weight of 10 heaviest images")]
for k, (values, name) in enumerate(plot_items):
    plt.subplot(1, 4, k + 1)
    plt.plot(range(1, n_rounds_used + 1), values)
    plt.xlabel("round")
    plt.title(name)
plt.tight_layout()
plt.savefig(OUT + "/q2b_plots.png", dpi=200)
plt.close()

# ==================================================================
# Q2 (c): boosting vs bagging with 25, 50, 100 learners
# ==================================================================
log("\n===== Q2(c) bagging vs boosting =====")
for L in [25, 50, 100]:
    start = time.time()
    t_bag, _, _ = train_bagging(X_train, y_train, L, best_depth, best_leaf, np.random.default_rng(SEED + 2))
    bag_time = time.time() - start

    start = time.time()
    m = train_samme(X_train, y_train, BOOST_DEPTH, L, NUM_CLASSES)
    boost_time = time.time() - start

    bag_accs, boost_accs = [], []
    for X_, y_ in [(X_train, y_train), (X_val, y_val), (X_test, y_test)]:
        bag_accs.append(accuracy(count_votes(trees_predict(t_bag, X_)).argmax(axis=1), y_))
        S_, q_ = samme_scores(samme_learner_preds(m, X_), m["alphas"], L, NUM_CLASSES)
        boost_accs.append(accuracy(q_.argmax(axis=1), y_))
    log("L=%3d | bagging train/val/test = %.3f/%.3f/%.3f time %.1fs | boosting train/val/test = %.3f/%.3f/%.3f time %.1fs (rounds used %d)"
        % (L, bag_accs[0], bag_accs[1], bag_accs[2], bag_time,
           boost_accs[0], boost_accs[1], boost_accs[2], boost_time, len(m["alphas"])))

# a training face that gets a very large boosting weight
max_weight_per_sample = boost["w_before"].max(axis=0)
heavy = int(np.argmax(max_weight_per_sample))
learner_preds_train = samme_learner_preds(boost, X_train)
was_correct = (learner_preds_train[:, heavy] == y_train[heavy])
S_train, q_train = samme_scores(learner_preds_train, boost["alphas"], n_rounds_used, NUM_CLASSES)
log("heaviest training face: index", heavy, "| true class", y_train[heavy],
    "| max weight %.4f (equal weight would be %.4f)" % (max_weight_per_sample[heavy], 1.0 / n_train))
log("  learners that predicted it correctly:", int(was_correct.sum()), "out of", n_rounds_used,
    "| final ensemble prediction:", int(S_train[heavy].argmax()))

plt.figure(figsize=(11, 3))
plt.subplot(1, 3, 1)
plt.imshow(X_train[heavy].reshape(64, 64), cmap="gray")
plt.title("training face %d, class %d" % (heavy, y_train[heavy]), fontsize=8)
plt.axis("off")
plt.subplot(1, 3, 2)
plt.semilogy(boost["w_before"][:, heavy])
plt.title("its weight in each round")
plt.xlabel("round")
plt.subplot(1, 3, 3)
plt.plot(was_correct.astype(int), ".")
plt.title("learner correct on it? (1 = yes)")
plt.xlabel("round")
plt.tight_layout()
plt.savefig(OUT + "/q2c_heavy_face.png", dpi=200)
plt.close()

# ==================================================================
# Q2 (d): tree vs bagging vs boosting on clean and masked test sets
# ==================================================================
log("\n===== Q2(d) clean vs masked =====")
bag_trees_final, _, _ = train_bagging(X_train, y_train, B_FINAL, best_depth, best_leaf, np.random.default_rng(SEED + 3))

results = {}        # results[(model, dataset)] = (accuracy, mean confidence)
store = {}          # keep predictions to find an interesting face later
for name, X_ in [("clean", X_test), ("masked", X_test_masked)]:
    proba = tree.predict_proba(X_)
    tree_pred = tree.classes_[proba.argmax(axis=1)]
    bag_preds = trees_predict(bag_trees_final, X_)
    bag_votes = count_votes(bag_preds)
    boost_S, boost_q = samme_scores(samme_learner_preds(boost, X_), boost["alphas"], n_rounds_used, NUM_CLASSES)
    results[("tree", name)] = (accuracy(tree_pred, y_test), proba.max(axis=1).mean())
    results[("bagging", name)] = (accuracy(bag_votes.argmax(axis=1), y_test), (bag_votes.max(axis=1) / B_FINAL).mean())
    results[("boosting", name)] = (accuracy(boost_q.argmax(axis=1), y_test), boost_q.max(axis=1).mean())
    store[name] = (bag_preds, bag_votes, boost_q)
for model_name in ["tree", "bagging", "boosting"]:
    log("%-9s clean: acc %.3f conf %.3f | masked: acc %.3f conf %.3f" %
        (model_name, results[(model_name, "clean")][0], results[(model_name, "clean")][1],
         results[(model_name, "masked")][0], results[(model_name, "masked")][1]))


def mask_split_fraction(trees):
    """fraction of split nodes that look at a pixel inside the masked block"""
    total, inside = 0, 0
    for t in trees:
        features = t.tree_.feature[t.tree_.children_left != -1]
        total += len(features)
        for f in features:
            if f in mask_pixels:
                inside += 1
    return inside / max(total, 1)


log("fraction of split nodes using masked pixels: tree %.3f | bagging %.3f | boosting %.3f" %
    (mask_split_fraction([tree]), mask_split_fraction(bag_trees_final), mask_split_fraction(boost["learners"])))

# find a face where bagging and boosting behave differently on the masked image
bag_votes_c, bag_votes_m = store["clean"][1], store["masked"][1]
boost_q_c, boost_q_m = store["clean"][2], store["masked"][2]
different = []
for i in range(len(y_test)):
    bag_ok = (bag_votes_m[i].argmax() == y_test[i])
    boost_ok = (boost_q_m[i].argmax() == y_test[i])
    if bag_ok != boost_ok:
        different.append(i)
if len(different) == 0:
    # fall back: face where the change in true-class score differs most between the two models
    best_diff, j = -1, 0
    for i in range(len(y_test)):
        d_bag = bag_votes_m[i, y_test[i]] / B_FINAL - bag_votes_c[i, y_test[i]] / B_FINAL
        d_boost = boost_q_m[i, y_test[i]] - boost_q_c[i, y_test[i]]
        if abs(d_bag - d_boost) > best_diff:
            best_diff, j = abs(d_bag - d_boost), i
else:
    j = different[0]
same_person_train = np.where(y_train == y_test[j])[0]
log("face", j, "(true class %d) | bagging true-class vote fraction clean %.2f -> masked %.2f (predicts %d on masked)" %
    (y_test[j], bag_votes_c[j, y_test[j]] / B_FINAL, bag_votes_m[j, y_test[j]] / B_FINAL, bag_votes_m[j].argmax()))
log("   boosting true-class q clean %.2f -> masked %.2f (predicts %d on masked)" %
    (boost_q_c[j, y_test[j]], boost_q_m[j, y_test[j]], boost_q_m[j].argmax()))
log("   pairwise disagreement of the bagged trees on the masked face: %.3f" %
    pairwise_disagreement(store["masked"][0][:, [j]]))
log("   average boosting weight of this person's 6 training faces: %.4f total (equal weight would be %.4f)" %
    (boost["w_before"].mean(axis=0)[same_person_train].sum(), 6.0 / n_train))

plt.figure(figsize=(9, 3))
plt.subplot(1, 3, 1)
plt.imshow(X_test[j].reshape(64, 64), cmap="gray")
plt.title("clean, true class %d" % y_test[j], fontsize=8)
plt.axis("off")
plt.subplot(1, 3, 2)
plt.imshow(X_test_masked[j].reshape(64, 64), cmap="gray")
plt.title("masked", fontsize=8)
plt.axis("off")
plt.subplot(1, 3, 3)
plt.bar(["bag clean", "bag masked", "boost clean", "boost masked"],
        [bag_votes_c[j, y_test[j]] / B_FINAL, bag_votes_m[j, y_test[j]] / B_FINAL,
         boost_q_c[j, y_test[j]], boost_q_m[j, y_test[j]]])
plt.xticks(fontsize=6)
plt.title("true-class vote / probability", fontsize=8)
plt.tight_layout()
plt.savefig(OUT + "/q2d_face.png", dpi=200)
plt.close()

# ==================================================================
# Q3: bias and variance
# ==================================================================
log("\n===== Q3 bias and variance (this part is slow) =====")
# indices of the 6 training faces of every person
person_rows = []
for c in range(NUM_CLASSES):
    person_rows.append(np.where(y_train == c)[0])
true_onehot = one_hot(y_test)


def bias_variance(all_p):
    """all_p has shape (repeats, test samples, classes)"""
    p_mean = all_p.mean(axis=0)
    bias2 = np.mean(np.sum((p_mean - true_onehot) ** 2, axis=1))
    variance = np.mean(np.sum((all_p - p_mean) ** 2, axis=2))
    accs = [accuracy(p.argmax(axis=1), y_test) for p in all_p]
    return bias2, variance, np.mean(accs), np.std(accs, ddof=1)


# depths for Q3(b): one shallow, the selected one and one much deeper
if best_depth is None:
    depths_b = [2, 8, None]
elif best_depth == 2:
    depths_b = [1, 2, None]
else:
    depths_b = [2, best_depth, None]
log("depths used in Q3(b):", depths_b)

p_tree, p_bag, p_boost = [], [], []                       # Q3(a)
pb_tree = [[] for d in depths_b]                          # Q3(b), one list per depth
pb_bag = [[] for d in depths_b]
pb_dis = [[] for d in depths_b]
pb_leaf_tree = [[] for d in depths_b]
pb_leaf_bag = [[] for d in depths_b]

for rep in range(N_REPEATS):
    start = time.time()
    rep_rng = np.random.default_rng(SEED + 100 + rep)
    # new training set: 6 faces sampled with replacement from each person's 6 faces
    chosen = []
    for c in range(NUM_CLASSES):
        chosen.extend(rep_rng.choice(person_rows[c], 6, replace=True))
    Xr, yr = X_train[chosen], y_train[chosen]

    # Q3(a): selected tree, bagging, boosting
    t = DecisionTreeClassifier(max_depth=best_depth, min_samples_leaf=best_leaf, random_state=0).fit(Xr, yr)
    p_tree.append(one_hot(t.predict(X_test)))
    trees_r, _, _ = train_bagging(Xr, yr, B_FINAL, best_depth, best_leaf, rep_rng)
    p_bag.append(count_votes(trees_predict(trees_r, X_test)) / B_FINAL)
    m = train_samme(Xr, yr, BOOST_DEPTH, BOOST_ROUNDS, NUM_CLASSES)
    p_boost.append(samme_scores(samme_learner_preds(m, X_test), m["alphas"], BOOST_ROUNDS, NUM_CLASSES)[1])

    # Q3(b): three depths, 100 tree bagging each
    for k in range(len(depths_b)):
        d = depths_b[k]
        t = DecisionTreeClassifier(max_depth=d, min_samples_leaf=best_leaf, random_state=0).fit(Xr, yr)
        trees_r, _, _ = train_bagging(Xr, yr, 100, d, best_leaf, rep_rng)
        preds = trees_predict(trees_r, X_test)
        pb_tree[k].append(one_hot(t.predict(X_test)))
        pb_bag[k].append(count_votes(preds) / 100)
        pb_dis[k].append(pairwise_disagreement(preds))
        pb_leaf_tree[k].append(len(Xr) / t.get_n_leaves())
        pb_leaf_bag[k].append(np.mean([len(Xr) / tt.get_n_leaves() for tt in trees_r]))
    log("repeat", rep + 1, "of", N_REPEATS, "done in %.0f s" % (time.time() - start))

# ---- Q3(a)
log("\n----- Q3(a) -----")
final_table = {}
for name, all_p in [("single tree", p_tree), ("bagging", p_bag), ("boosting", p_boost)]:
    b2, var, acc_mean, acc_std = bias_variance(np.array(all_p))
    final_table[name] = (b2, var)
    log("%-12s bias^2 %.4f  variance %.4f | test acc over %d refits: mean %.3f std %.3f" %
        (name, b2, var, N_REPEATS, acc_mean, acc_std))

# ---- Q3(b)
log("\n----- Q3(b) -----")
for k in range(len(depths_b)):
    b2_t, var_t, acc_t, _ = bias_variance(np.array(pb_tree[k]))
    b2_b, var_b, acc_b, _ = bias_variance(np.array(pb_bag[k]))
    removed = var_t - var_b
    log("depth", depths_b[k], "| tree: bias^2 %.4f var %.4f acc %.3f | bagging(100): bias^2 %.4f var %.4f acc %.3f" %
        (b2_t, var_t, acc_t, b2_b, var_b, acc_b))
    log("    variance removed by bagging: %.4f (%.1f%%) | accuracy gain %.3f | samples per leaf: tree %.2f, bagged trees %.2f | tree disagreement %.3f" %
        (removed, 100 * removed / max(var_t, 1e-12), acc_b - acc_t,
         np.mean(pb_leaf_tree[k]), np.mean(pb_leaf_bag[k]), np.mean(pb_dis[k])))

# ---- Q3(d): final table
log("\n----- Q3(d) final table -----")
log("%-12s %12s %12s %10s %10s" % ("model", "clean acc", "masked acc", "bias^2", "variance"))
for model_name, key in [("single tree", "tree"), ("bagging", "bagging"), ("boosting", "boosting")]:
    log("%-12s %12.3f %12.3f %10.4f %10.4f" % (model_name, results[(key, "clean")][0],
                                              results[(key, "masked")][0],
                                              final_table[model_name][0], final_table[model_name][1]))

plt.figure(figsize=(4.5, 3))
names = ["single tree", "bagging", "boosting"]
plt.bar(np.arange(3) - 0.2, [final_table[n][0] for n in names], 0.4, label="bias^2")
plt.bar(np.arange(3) + 0.2, [final_table[n][1] for n in names], 0.4, label="variance")
plt.xticks(range(3), names, fontsize=8)
plt.legend()
plt.tight_layout()
plt.savefig(OUT + "/q3_bias_variance.png", dpi=200)
plt.close()

log("\nAll done. Results are in the folder:", OUT)
logfile.close()
