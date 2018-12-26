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
        if cr(i,j)>140 && cr(i,j)<160 %��ȡ�з�ɫ����
            skin(i,j) = 1;
        end
    end
end
figure;
imshow(skin, []);
title('��ɫ������');

mask = strel('disk', 6); %��������̬ѧ����
skin_closed = imclose(skin, mask); %����ȡ�ķ�ɫ������б�����
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
title('�����������ķ�ɫ����');
L = bwlabel(skin_closed); %ͳ���з�ɫ����
foreground = regionprops(L, 'area'); %�����з�ɫ�������
fg_areas = [foreground.Area];
fg_max = max(fg_areas); %ȡ������ķ�ɫ����
threshold = 27;
if fg_max>m*n/threshold
    face_candidate = bwareaopen(skin_closed, fg_max); %ɾ�����С��fg_max������
    figure;
    imshow(face_candidate);
    title('ɾ��С��������');
    %�ҳ���������Χ
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
    %��������������
    if region_h>.5*region_w && region_h<2*region_w
        %ȡ�����Ķ�ͷ���֣�ʹ������С������
        upper_face = imcrop(face_candidate, [x_start y_start region_w .4*region_h]);
        figure;
        imshow(upper_face);
        title('��ͷ����');
        [m1, n1] = size(upper_face);
        L1 = bwlabel(upper_face);
        up_fa = regionprops(L1, 'area');
        uf_areas = [up_fa.Area];
        uf_max = max(uf_areas);
        if uf_max/(m1*n1)>.3
            upface = bwareaopen(upper_face, floor(.5*uf_max));
            x1 = n1;
            x2 = n1;
            %�����ͷ���
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