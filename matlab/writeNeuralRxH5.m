function writeNeuralRxH5(filename, data)
%WRITENEURALRXH5 Write a batch using the neural receiver HDF5 contract.
%
% Required fields (logical MATLAB dimensions):
%   rxGrid       [N,S,K,2] complex
%   dataMask     [N,S,K,2] logical
%   dmrsGrid     [N,S,K,2] complex, known transmitted pilots, zero elsewhere
%   dmrsMask     [N,S,K,2] logical, same layer/port convention as txSymbols
%   mcsIndex     [N,1], qm [N,1], codeRate [N,1]
%   numRBs       [N,1], noiseVar [N,1]
% Optional metadata: snrDb [N,1]
% Legacy diagnostic field (not read by the neural receiver):
%   channelEst   [N,S,K,2,2] complex, H(rx,layer)
% Training files additionally require:
%   txSymbols    [N,S,K,2] complex
%   codedBits    [N,S,K,2,Qmax] single (0/1; Qmax <= 10)
% Optional training comparison field:
%   traditionalEq [N,S,K,2] complex

arguments
    filename (1,:) char
    data struct
end

if isfile(filename)
    delete(filename);
end
writeReal(filename, '/rx_grid', complexToIQ(data.rxGrid));
h5writeatt(filename, '/', 'axis_order', 'matlab');
h5writeatt(filename, '/', 'format_version', int32(2));
if isfield(data, 'channelEst')
    writeReal(filename, '/channel_est', complexToIQ(data.channelEst));
end
writeReal(filename, '/data_mask', uint8(data.dataMask));
writeReal(filename, '/dmrs_grid', complexToIQ(data.dmrsGrid));
writeReal(filename, '/dmrs_mask', uint8(data.dmrsMask));

requiredMetadata = {'mcsIndex','qm','codeRate','numRBs','noiseVar'};
storedNames = {'mcs_index','qm','code_rate','num_rbs','noise_var'};
for index = 1:numel(requiredMetadata)
    name = requiredMetadata{index};
    assert(isfield(data, name), 'Missing metadata field %s.', name);
    writeReal(filename, ['/' storedNames{index}], data.(name)(:));
end
if isfield(data, 'snrDb')
    writeReal(filename, '/snr_db', single(data.snrDb(:)));
end

if isfield(data, 'txSymbols')
    writeReal(filename, '/tx_symbols', complexToIQ(data.txSymbols));
end
if isfield(data, 'codedBits')
    writeReal(filename, '/coded_bits', single(data.codedBits));
end
if isfield(data, 'traditionalEq')
    writeReal(filename, '/traditional_eq', complexToIQ(data.traditionalEq));
end
end

function value = complexToIQ(value)
value = cat(ndims(value) + 1, single(real(value)), single(imag(value)));
end

function writeReal(filename, dataset, value)
value = gather(value);
h5create(filename, dataset, size(value), 'Datatype', class(value), ...
    'ChunkSize', chooseChunk(size(value)));
h5write(filename, dataset, value);
end

function chunk = chooseChunk(shape)
chunk = shape;
chunk(1) = 1;
% Keep chunks below roughly 8 MiB without changing sample alignment.
while prod(double(chunk)) * 4 > 8 * 1024 * 1024
    [~, index] = max(chunk(2:end));
    index = index + 1;
    chunk(index) = max(1, ceil(chunk(index) / 2));
end
end
