%% Basic MATLAB -> PyTorch joint neural receiver call
% Run after OFDM demodulation. No traditional channel estimate/equalizer is used.

projectDir = fileparts(fileparts(mfilename('fullpath')));
checkpoint = fullfile(projectDir, 'runs', 'default', 'best.pt');
inputFile = fullfile(tempdir, 'neural_rx_input.h5');
outputFile = fullfile(tempdir, 'neural_rx_output.mat');

% Replace these example variables with receiver outputs. Array convention is
% [batch, OFDM symbol, allocated subcarrier, RX antenna/layer].
N = 1; S = 14; numRBs = 10; K = 12 * numRBs;
rxGrid = complex(zeros(N,S,K,2,'single'));             % replace with received grid
dmrsGrid = complex(zeros(N,S,K,2,'single'));           % known TX DM-RS per layer
dmrsMask = false(N,S,K,2);                             % replace with actual DM-RS mask
% Runnable toy comb pilots; replace both lines with nrPDSCHDMRS outputs.
dmrsMask(:,3,1:2:end,1) = true;
dmrsMask(:,3,2:2:end,2) = true;
dmrsGrid(dmrsMask) = complex(single(1), single(0));
dataMask = true(N,S,K,2);
dataMask(dmrsMask) = false;                            % also clear PTRS/unused REs
mcsIndex = int32(10);                                  % your selected MCS
qm = int32(4);                                         % modulation order
codeRate = single(490 / 1024);                         % target code rate
noiseVar = single(0.01);                               % E[|n|^2], complex

request = struct('rxGrid',rxGrid, 'dmrsGrid',dmrsGrid, ...
    'dmrsMask',dmrsMask, 'dataMask',dataMask, ...
    'mcsIndex',mcsIndex, 'qm',qm, ...
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
neuralChannel = result.refined_channel;                 % [N,S,K,rx,layer]

fprintf('Received %d neural equalized constellation points.\n', ...
    nnz(result.data_mask));
