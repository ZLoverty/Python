function cross_correlation
main_test
% idx = 1:6;
% him = 2;
% wim = 3;
% new_idx = convert_index(idx, him, wim)
end

function main_test
% img = imread('C:\Users\liux3141\Desktop\2i.tif');
mask = imread('C:\Users\liux3141\Desktop\maski.tif');
imgstack = 'C:\Users\liux3141\Desktop\video.tif'
info = imfinfo(imgstack);
num_images = numel(info);
number_of_particle = 3;
fig = figure(1);
ax = axes(fig);
saveDir = '13particle\';
fid = fopen([saveDir 'xyt.dat'], 'w');
fprintf(fid, '%-7s\t%-7s\t%-7s\n', 'X', 'Y', 'Frame');
for i = 1: num_images
    img_ori = imread(imgstack, i);
    img = imcomplement(img_ori);
    [max_coor, pk_value] = track_spheres(img, mask, number_of_particle)
    
%     imagesc(img_ori);
%     colormap(gray);
%     hold on
%     plot3(max_coor(1, :), max_coor(2,:), pk_value, 'LineStyle', 'None', 'Marker', 'o',...
%     'MarkerFaceColor', 'None', 'MarkerEdgeColor', 'r', 'MarkerSize', 30, 'LineWidth', 5);
%     hold off
%     axis off
%     pause(.001);
%     F = getframe(ax);
%     Image = frame2im(F);
%     imwrite(Image, [saveDir num2str(i) '.jpg']);
%     for j = 1: number_of_particle
%         fprintf(fid, '%3.3f\t%3.3f\t%3.3f\n', [max_coor(1, j) max_coor(2,j) i])
%     end
end

% c = normxcorr2(mask, img);
% [him, wim] = size(img);
% [hma, wma] = size(mask);
% ori_r = floor(hma/2);
% ori_c = floor(wma/2);
% c_crop = c(ori_r: ori_r+him-1, ori_c: ori_c+wim-1);
% surf(c_crop), shading flat
% cent=FastPeakFind(c_crop);
% cent2 = reshape(cent, [2, numel(cent)/2]);
% peaks = zeros(1, numel(cent)/2);
% for i = 1: numel(cent)/2
%     row = floor(cent2(2, i));
%     col = floor(cent2(1, i));
%     a = c_crop(row, col);
%     peaks(1, i) = a;
% end

end

function [max_coor, pk_value] = track_spheres(img, mask, number_of_particle)
c = normxcorr2(mask, img);
[him, wim] = size(img);
[hma, wma] = size(mask);
ori_r = floor(hma/2);
ori_c = floor(wma/2);
c_crop = c(ori_r: ori_r+him-1, ori_c: ori_c+wim-1);
cent=FastPeakFind(c_crop);
cent2 = reshape(cent, [2, numel(cent)/2]);
peaks = zeros(1, numel(cent)/2);
for i = 1: numel(cent)/2
    row = floor(cent2(2, i));
    col = floor(cent2(1, i));
    a = c_crop(row, col);
    peaks(1, i) = a;
end
[pks, index] = maxk(peaks, number_of_particle);
max_coor = zeros(2, number_of_particle);
pk_value = zeros(1, number_of_particle);
for i = 1: number_of_particle
    max_coor(:, i) = cent2(:, index(i));
    pk_value(1, i) = pks(i);
end

end

function new_idx = convert_index(idx, him, wim)
new_idx = zeros(numel(idx), 2);
for i = 1: numel(idx)
    if mod(idx(i), him) == 0
        new_idx_row = him;
        new_idx_col = idx(i)/him;
    else
        new_idx_col = floor(idx(i)/him)+1;
        new_idx_row = mod(idx(i), him);
    end
    new_idx(i, 1) = new_idx_row;
    new_idx(i, 2) = new_idx_col;
end
end
