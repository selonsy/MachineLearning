clear all;
close all;
clc;
I = imread('face9.jpg');
ycbcr_im = rgb2ycbcr(I);
cr = ycbcr_im(:,:,3);
[m, n] = size(cr);
skin = zeros(m, n);
for i = 1:m
    for j = 1:n
        if cr(i,j)>140 && cr(i,j)<160 %提取有肤色区域
            skin(i,j) = 1;
        end
    end
end
figure;
imshow(skin, []);
title('肤色区域检测');

mask = strel('disk', 6); %闭运算形态学算子
skin_closed = imclose(skin, mask); %对提取的肤色区域进行闭运算
%{
skin_opened = imopen(skin, mask);
figure;
imshow(skin_opened, []);
title('open');
mask2 = strel('disk', 15);
skin_closed = imclose(skin_opened, mask); 
%}
figure;
imshow(skin_closed, []);
title('经过闭运算后的肤色区域');
L = bwlabel(skin_closed); %统计有肤色区域
foreground = regionprops(L, 'area'); %计算有肤色区域面积
fg_areas = [foreground.Area];
fg_max = max(fg_areas); %取面积最大的肤色区域
threshold = 27;
if fg_max>m*n/threshold
    face_candidate = bwareaopen(skin_closed, fg_max); %删除面积小于fg_max的区域
    figure;
    imshow(face_candidate);
    title('删除小面积区域后');
    %找出脸部区域范围
    y_start = m;
    y_end = m;
    x_start = n;
    x_end = n;
    for i = 1:m
        if any(face_candidate(i,:))==1
            y_start = i;
            break
        end
    end
    for i = y_start:m
        if any(face_candidate(i,:))==0
            y_end = i;
            break
        end
    end
    for j = 1:n
        if any(face_candidate(:,j))==1
            x_start = j;
            break
        end
    end
    for j = x_start:n
        if any(face_candidate(:,j))==0
            x_end = j;
            break
        end
    end
    region_h = y_end - y_start;
    region_w = x_end - x_start;
    %画框标出人脸区域
    if region_h>.5*region_w && region_h<2*region_w
        %取脸部的额头部分，使框尽量缩小到脸部
        upper_face = imcrop(face_candidate, [x_start y_start region_w .4*region_h]);
        figure;
        imshow(upper_face);
        title('额头部分');
        [m1, n1] = size(upper_face);
        L1 = bwlabel(upper_face);
        up_fa = regionprops(L1, 'area');
        uf_areas = [up_fa.Area];
        uf_max = max(uf_areas);
        if uf_max/(m1*n1)>.3
            upface = bwareaopen(upper_face, floor(.5*uf_max));
            x1 = n1;
            x2 = n1;
            %计算额头宽度
            for j = 1:n1
                if any(upface(:,j))==1
                    x1 = j;
                    break
                end
            end
            for j = x1:n1
                if any(upface(:,j))==0
                    x2 = j;
                    break
                end
            end
            fw = x2-x1;
            figure;
            imshow(I);
            title('flag');
            hold on
            h1 = line([x_start+x1, x_start+x1+fw], [y_start, y_start]);
            h2 = line([x_start+x1+fw, x_start+x1+fw], [y_start, y_start+1.2*fw]);
            h3 = line([x_start+x1+fw, x_start+x1], [y_start+1.2*fw, y_start+1.2*fw]);
            h4 = line([x_start+x1, x_start+x1], [y_start+1.2*fw, y_start]);
            h = [h1 h2 h3 h4];
            set(h, 'Color', [1 0 0], 'LineWidth', 2);
        else
            figure;
            imshow(I);
        end
    else
       figure;
       imshow(I);
    end
else
    figure;
    imshow(I);
end