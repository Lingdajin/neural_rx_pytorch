%% Basic MATLAB -> PyTorch neural receiver call
% Run after OFDM demodulation and noise estimation. Keep received pilot REs.

projectDir = fileparts(fileparts(mfilename('fullpath')));
checkpoint = fullfile(projectDir, 'runs', 'no_csi', 'best.pt');
inputFile = fullfile(tempdir, 'neural_rx_input.h5');
outputFile = fullfile(tempdir, 'neural_rx_output.mat');

% Replace these example variables with receiver outputs. Array convention is
% [batch, OFDM symbol, allocated subcarrier, RX antenna/layer].
N = 1; S = 14; numRBs = 10; K = 12 * numRBs;
rxGrid = complex(zeros(N,S,K,2,'single'));             % received grid
dataMask = true(N,S,K,2);                              % false on DM-RS/PTRS
mcsIndex = int32(10);                                  % your selected MCS
qm = int32(4);                                         % modulation order
codeRate = single(490 / 1024);                         % target code rate
noiseVar = single(0.01);                               % E[|n|^2], complex

request = struct('rxGrid',rxGrid, ...
    'dataMask',dataMask, 'mcsIndex',mcsIndex, 'qm',qm, ...
    'codeRate',codeRate, 'numRBs',int32(numRBs), 'noiseVar',noiseVar);
writeNeuralRxH5(inputFile, request);

pythonExe = 'python';  % Set an absolute path to the environment with PyTorch.
command = sprintf('"%s" "%s" --input "%s" --checkpoint "%s" --output "%s"', ...
    pythonExe, fullfile(projectDir, 'infer.py'), inputFile, checkpoint, outputFile);
[status, message] = system(command);
assert(status == 0, 'Neural receiver failed: %s', message);
result = load(outputFile);

% Positive LLR means bit=1. Select only scheduled data REs and bits 1:qm.
neuralSymbols = result.equalized_symbols;              % [N,S,K,layer]
bitLLR = result.bit_llr_positive_one(:,:,:,:,1:double(qm));

fprintf('Received %d neural equalized constellation points.\n', ...
    nnz(result.data_mask));
