// package at.ac.tuwien.big.moea.search.algorithm.reinforcement.networks;
//
// import at.ac.tuwien.big.moea.search.algorithm.provider.AbstractRegisteredAlgorithm;
//
// import java.nio.file.Paths;
//
// public static class DQNNBuilder<S> {
//
// private double gamma;
// private double lr;
// private double eps;
// private int framesPerEpoch;
// private int bufferSize;
// private int batchSize;
// private int updateTarget;
// private double l2Regularization;
// private double dropoutRate;
// private String networkPath;
// private boolean disableRegularization;
// private String networkSavePath;
// private String scoreSavePath;
// private int epochsPerModelSave;
// private boolean enableProgressServer;
// private int terminateAfterEpisodes;
// private boolean verbose;
//
// public DQNNBuilder() {
// }
//
// DQNNBuilder(final double gamma, final double lr, final double eps, final int framesPerEpoch, final int bufferSize,
// final int batchSize, final int updateTarget, final double l2Regularization, final double dropoutRate, final String
// networkPath, final boolean disableRegularization, final String networkSavePath, final String scoreSavePath, final int
// epochsPerModelSave, final boolean enableProgressServer, final int terminateAfterEpisodes, final boolean verbose) {
// this.gamma = gamma;
// this.lr = lr;
// this.eps = eps;
// this.framesPerEpoch = framesPerEpoch;
// this.bufferSize = bufferSize;
// this.batchSize = batchSize;
// this.updateTarget = updateTarget;
// this.l2Regularization = l2Regularization;
// this.dropoutRate = dropoutRate;
// this.networkPath = networkPath;
// this.disableRegularization = disableRegularization;
// this.networkSavePath = networkSavePath;
// this.scoreSavePath = scoreSavePath;
// this.epochsPerModelSave = epochsPerModelSave;
// this.enableProgressServer = enableProgressServer;
// this.terminateAfterEpisodes = terminateAfterEpisodes;
// this.verbose = verbose;
// }
//
// p
// public DQNNBuilder batchSize(final int batchSize) {
// this.batchSize = batchSize;
// return DQNNBuilder.this;
// }
//
// public DQNNBuilder bufferSize(final int bufferSize) {
// this.bufferSize = bufferSize;
// return DQNNBuilder.this;
// }
//
// public AbstractRegisteredAlgorithm<EpsGreedyNetworkAgent<S>> build() {
//
// return new AbstractRegisteredAlgorithm<EpsGreedyNetworkAgent<S>>() {
// @Override
// public EpsGreedyNetworkAgent<S> createAlgorithm() {
// return new EpsGreedyNetworkAgent<>(networkAgent, dqTargetNet, eps, framesPerEpoch, bufferSize, batchSize,
// updateTarget, createProblem(), getPolicyEnvironment(),
// scoreSavePath != null ? Paths.get(getOutputPath(), "rewards", scoreSavePath).toString() : null,
// epochsPerModelSave, terminateAfterEpisodes, verbose);
// }
// };
// }
//
// public DQNNBuilder disableRegularization(final boolean disableRegularization) {
// this.disableRegularization = disableRegularization;
// return DQNNBuilder.this;
// }
//
// public DQNNBuilder dropoutRate(final double dropoutRate) {
// this.dropoutRate = dropoutRate;
// return DQNNBuilder.this;
// }
//
// public DQNNBuilder enableProgressServer(final boolean enableProgressServer) {
// this.enableProgressServer = enableProgressServer;
// return DQNNBuilder.this;
// }
//
// public DQNNBuilder epochsPerModelSave(final int epochsPerModelSave) {
// this.epochsPerModelSave = epochsPerModelSave;
// return DQNNBuilder.this;
// }
//
// public DQNNBuilder eps(final double eps) {
// this.eps = eps;
// return DQNNBuilder.this;
// }
//
// public DQNNBuilder framesPerEpoch(final int framesPerEpoch) {
// this.framesPerEpoch = framesPerEpoch;
// return DQNNBuilder.this;
// }
//
// public DQNNBuilder gamma(final double gamma) {
// this.gamma = gamma;
// return DQNNBuilder.this;
// }
//
// public DQNNBuilder l2Regularization(final double l2Regularization) {
// this.l2Regularization = l2Regularization;
// return DQNNBuilder.this;
// }
//
// public DQNNBuilder lr(final double lr) {
// this.lr = lr;
// return DQNNBuilder.this;
// }
//
// public DQNNBuilder networkPath(final String networkPath) {
// this.networkPath = networkPath;
// return DQNNBuilder.this;
// }
//
// public DQNNBuilder networkSavePath(final String networkSavePath) {
// this.networkSavePath = networkSavePath;
// return DQNNBuilder.this;
// }
//
// public DQNNBuilder scoreSavePath(final String scoreSavePath) {
// this.scoreSavePath = scoreSavePath;
// return DQNNBuilder.this;
// }
//
// public DQNNBuilder terminateAfterEpisodes(final int terminateAfterEpisodes) {
// this.terminateAfterEpisodes = terminateAfterEpisodes;
// return DQNNBuilder.this;
// }
//
// public DQNNBuilder updateTarget(final int updateTarget) {
// this.updateTarget = updateTarget;
// return DQNNBuilder.this;
// }
//
// public DQNNBuilder verbose(final boolean verbose) {
// this.verbose = verbose;
// return DQNNBuilder.this;
// }
// }
