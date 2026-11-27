function match_source_frames(run, archive, refine)
% Find provenance candidates by similarity of the supplied crops to decoded video.
if nargin < 3, refine = false; end
records = readtable(fullfile(run,'intake.csv'), TextType="string");
originals = fullfile(archive,'originals');
if refine
    coarse = readtable(fullfile(run,'feature_candidates.csv'),TextType="string");
    out = fullfile(run,'matches');
    names = [fields(), {'ncc_gray','ncc_texture','similarity','decoder_time_s'}];
else
    out = fullfile(run,'candidates');
    names = fields();
end
if ~isfolder(out), mkdir(out); end
summary = cell(0,numel(names));
for row = 1:height(records)
    ref = records.reference(row);
    video = records.source_video(row);
    input = fullfile(originals, video);
    reader = VideoReader(input);
    if ismissing(records.attachment(row))
        fprintf('%s %s %.3fs %.8f fps\n',ref,video,reader.Duration,reader.FrameRate);
        continue
    end
    snap = imread(fullfile(run,'attachments',ref+".png"));
    if size(snap,3) > 3, snap = snap(:,:,1:3); end
    ts = records.user_time_s(row);
    fps = reader.FrameRate;
    % Playback time was supplied at one-second precision.
    first = max(0,floor((ts-1.05)*fps));
    last = min(floor(reader.Duration*fps)-1,ceil((ts+1.60)*fps));
    if last < first
        error('Timestamp outside video for %s',ref);
    end
    [queryFeatures, queryPts, queryCorners, queryImage] = screenshotFeatures(snap);
    if refine
        candidates = coarse(coarse.reference==ref,:);
        % Refine every frame close to the best feature matches in several parts of the clip.
        chosen = candidates.frame(1:min(8,height(candidates)));
        targetFrames = unique(reshape(chosen(:)+(-6:6),[],1));
        targetFrames = targetFrames(targetFrames>=0 & targetFrames<=last);
        last = max(targetFrames);
        first = min(targetFrames);
    end
    tested = cell(0,numel(names));
    rgb = cell(0,1);
    frame = 0;
    while hasFrame(reader) && frame <= last
        decoderTime = reader.CurrentTime;
        decoded = readFrame(reader);
        if (refine && ismember(frame,targetFrames)) || ...
                (~refine && frame>=first && (mod(frame-first,5)==0 || frame==round(ts*fps)))
            if refine
                result = matchOne(decoded,queryFeatures,queryPts,queryCorners,queryImage);
                if ~isempty(result), result = [result,{decoderTime}]; end
            else
                result = matchOne(decoded,queryFeatures,queryPts,queryCorners);
            end
            if ~isempty(result)
                tested(end+1,:) = {char(ref),char(video),frame,frame/fps,result{:}}; %#ok<AGROW>
                rgb{end+1} = decoded; %#ok<AGROW>
            end
        end
        frame = frame+1;
    end
    if isempty(tested)
        fprintf('%s no feature-match; searched %d..%d (duration %.2fs)\n',ref,first,last,reader.Duration);
    else
        matches = cell2table(tested, 'VariableNames', names);
        if refine
            [ranked,ord] = sortrows(matches,{'similarity','inliers'},{'descend','descend'});
        else
            [ranked,ord] = sortrows(matches,{'inliers','median_residual'},{'descend','ascend'});
        end
        keep = min(5,height(ranked));
        if refine, keep = 1; end
        for k = 1:keep
            name = sprintf('%s_%s_f%06d',ref, erase(video,'.MP4'),ranked.frame(k));
            imwrite(rgb{ord(k)},fullfile(out,name+".png"));
            if refine
                pix = rgb{ord(k)};
                x = max(0,floor(ranked.x(k)));
                y = max(0,floor(ranked.y(k)));
                right = min(size(pix,2),ceil(ranked.x(k)+ranked.w(k)+1));
                bottom = min(size(pix,1),ceil(ranked.y(k)+ranked.h(k)+1));
                imwrite(pix(y+1:bottom,x+1:right,:),fullfile(out,name+"_crop.png"));
            end
        end
        fprintf('%s best frame %d, %d inliers; searched %d..%d\n',ref,ranked.frame(1),ranked.inliers(1),first,last);
        summary = [summary; table2cell(ranked)]; %#ok<AGROW>
    end
end
name = 'feature_candidates.csv';
if refine, name = 'refined_candidates.csv'; end
writetable(cell2table(summary,'VariableNames',names), fullfile(run,name));
end

function names = fields()
names = {'reference','source_video','frame','nominal_s','inliers','pairs','median_residual','x','y','w','h'};
end

function [des,valid,corners,work] = screenshotFeatures(snap)
gray = im2gray(snap);
s = min(1,850/size(gray,2));
work = imresize(gray,s);
corners = [1,1; size(work,2),1; size(work,2),size(work,1); 1,size(work,1)];
points = detectSIFTFeatures(work,ContrastThreshold=0.007);
[des,valid] = extractFeatures(work,points.selectStrongest(1200));
end

function result = matchOne(rgb,des,queryPts,corners,queryImage)
result = {};
scale = 0.5;
gray = imresize(im2gray(rgb),scale);
p = detectSIFTFeatures(gray,ContrastThreshold=0.009);
[features,valid] = extractFeatures(gray,p.selectStrongest(1800));
if isempty(features) || isempty(des), return; end
pairs = matchFeatures(des,features,MaxRatio=0.72,Unique=true,MatchThreshold=60);
if size(pairs,1) < 5, return; end
a = queryPts.Location(pairs(:,1),:);
b = valid.Location(pairs(:,2),:);
[transform,inliers,status] = estgeotform2d(a,b,"similarity",MaxDistance=2.5,MaxNumTrials=650);
if status~=0 || nnz(inliers)<5, return; end
warped = (transformPointsForward(transform,corners)-0.5)/scale+0.5;
minCorner = min(warped,[],1)-1;
boundingExtent = max(warped,[],1)-min(warped,[],1);
if any(boundingExtent < 10) || any(minCorner < -35) || any(minCorner+boundingExtent > [1920,1080]+35)
    return
end
estimated = transformPointsForward(transform,a(inliers,:));
residual = vecnorm(estimated-b(inliers,:),2,2);
result = {nnz(inliers),size(pairs,1),median(residual),minCorner(1),minCorner(2),boundingExtent(1),boundingExtent(2)};
if nargin >= 5
    [qx,qy] = meshgrid(1:2:size(queryImage,2),1:2:size(queryImage,1));
    sample = transformPointsForward(transform,[qx(:),qy(:)]);
    local = reshape(interp2(im2single(gray),sample(:,1),sample(:,2),'linear',NaN),size(qx));
    reference = im2single(queryImage(1:2:end,1:2:end));
    validMask = isfinite(local);
    if nnz(validMask)<0.80*numel(validMask), result={}; return; end
    local(~validMask) = reference(~validMask);
    tone = corrcoef(local(validMask),reference(validMask));
    a = local-imgaussfilt(local,1.25);
    b = reference-imgaussfilt(reference,1.25);
    mask = imerode(validMask,ones(5));
    texture = corrcoef(a(mask),b(mask));
    score = 0.45*tone(1,2)+0.55*texture(1,2);
    if ~isfinite(score), result={}; return; end
    result = [result,{tone(1,2),texture(1,2),score}];
end
end
