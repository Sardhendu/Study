function ret = a4_rand(requested_size, seed)
    global randomness_source
    start_i = mod(round(seed), round(size(randomness_source, 2) / 10)) + 1;
    if start_i + prod(requested_size) >= size(randomness_source, 2) + 1, 
        error('a4_rand failed to generate an array of that size (too big)')
    end
    ret = reshape(randomness_source(start_i : start_i+prod(requested_size)-1), requested_size);
end




# The network parameters are stored in  amatrix of [numHidUnits, numVisibleUnits]
# Units in an RBM are Binary , either 1 or 0