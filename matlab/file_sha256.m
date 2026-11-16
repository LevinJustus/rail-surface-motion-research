function hash = file_sha256(path)
%FILE_SHA256 Hash actual file bytes, without text newline conversion.
fid=fopen(path,'rb');assert(fid>=0,'Cannot read hash input: %s',path);
c=onCleanup(@()fclose(fid));
md=java.security.MessageDigest.getInstance('SHA-256');
while ~feof(fid)
    bytes=fread(fid,1024*1024,'*uint8');
    if ~isempty(bytes),md.update(typecast(bytes,'int8'));end
end
digest=typecast(md.digest(),'uint8');
hash=lower(reshape(dec2hex(digest,2).',1,[]));
end
