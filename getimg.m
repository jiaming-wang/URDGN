strh = strcat('./logs/YTC_example_v1_723.png'); %% the test image obtained from the last epoch
im = imread(strh);
im_l1 = im2double(im);
for i =1:4
    for j = 1:10
        img = im_l1((i-1)*144+1:i*144,(j-1)*144+1:j*144,:);
        strh1 = strcat('./h\',num2str((i-1)*10+j),'.png');
        imwrite(uint8(img*255 ),strh1,'png');
    end
end
