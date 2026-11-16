function hash=network_parameter_sha256(net)
%NETWORK_PARAMETER_SHA256 Fingerprint actual CPU single parameter bytes.
md=java.security.MessageDigest.getInstance('SHA-256');
for k=1:height(net.Learnables)
    name=[char(net.Learnables.Layer(k)) '/' char(net.Learnables.Parameter(k))];
    md.update(typecast(unicode2native(name,'UTF-8'),'int8'));
    value=extractdata(net.Learnables.Value{k});
    assert(isa(value,'single') && all(isfinite(value),'all'));
    md.update(typecast(value(:),'int8'));
end
hash=lower(reshape(dec2hex(typecast(md.digest(),'uint8'),2).',1,[]));
end
