clear;
close all;
clc;

imageFiles = {'6.png', '2.png', '3.png'};

randomIndex = randi([1, 3]); 
randomImage = imageFiles{randomIndex}; 

I = imread(randomImage); 

faceDetect = vision.CascadeObjectDetector();
bbox = step(faceDetect, I);

if ~isempty(bbox) 
    bbox = bbox(1, :); 
    face = imcrop(I, bbox);

    % detección de ojos, nariz y boca
    eyeDetect = vision.CascadeObjectDetector('RightEye');
    eyebox = step(eyeDetect, face);
    if size(eyebox, 1) >= 2
        eye1 = eyebox(1, :);  
        eye2 = eyebox(2, :); 
    else
        disp('No se detectaron ambos ojos.');
        return;
    end

    ndetect = vision.CascadeObjectDetector('Nose', 'MergeThreshold', 16);
    nosebox = step(ndetect, face);
    if ~isempty(nosebox)
        nose = nosebox(1, :); 
    else
        disp('No se detectó la nariz.');
        return;
    end

    mouthDetect = vision.CascadeObjectDetector('Mouth', 'MergeThreshold', 20);
    mouth = imcrop(face, [1, nose(2), size(face, 2), size(face, 1) - nose(2)]);
    mouthbox = step(mouthDetect, mouth);
    if ~isempty(mouthbox)
        mouthbox(1, 2) = mouthbox(1, 2) + nose(2); 
        mouth = mouthbox(1, :);
    else
        disp('No se detectó la boca.');
        return;
    end

    
    eye1(1:2) = eye1(1:2) + bbox(1:2);
    eye2(1:2) = eye2(1:2) + bbox(1:2);
    nose(1:2) = nose(1:2) + bbox(1:2);
    mouth(1:2) = mouth(1:2) + bbox(1:2);

    % Colores 
    colors = {'red', 'yellow', 'green', 'cyan', 'blue', 'magenta'};
    shape = [eye1(1) + eye1(3)/2, eye1(2) + eye1(4)/2; ...
             eye2(1) + eye2(3)/2, eye2(2) + eye2(4)/2; ...
             nose(1) + nose(3)/2, nose(2) + nose(4)/2; ...
             mouth(1) + mouth(3)/2, mouth(2) + mouth(4)/2];
    
   
    for idx = 1:length(colors)
        color = colors{idx};
        
        % Capa de color
        coloredImage = I;
        faceMask = false(size(I, 1), size(I, 2));
        faceMask(bbox(2):(bbox(2)+bbox(4)-1), bbox(1):(bbox(1)+bbox(3)-1)) = true;
        
        colorLayer = zeros(size(I), 'like', I);
        switch color
            case 'red'
                colorLayer(:,:,1) = 255;
            case 'yellow'
                colorLayer(:,:,1) = 255; colorLayer(:,:,2) = 255;
            case 'green'
                colorLayer(:,:,2) = 255;
            case 'cyan'
                colorLayer(:,:,2) = 255; colorLayer(:,:,3) = 255;
            case 'blue'
                colorLayer(:,:,3) = 255;
            case 'magenta'
                colorLayer(:,:,1) = 255; colorLayer(:,:,3) = 255;
        end
        
        % Transparencia de color
        alpha = 0.5;
        for c = 1:3
            channel = coloredImage(:,:,c);
            colorChannel = colorLayer(:,:,c);
            channel(faceMask) = uint8(alpha * double(colorChannel(faceMask)) + (1 - alpha) * double(channel(faceMask)));
            coloredImage(:,:,c) = channel;
        end
        

        imshow(coloredImage);
        hold on;
        plot(shape(:,1), shape(:,2), '+', 'MarkerSize', 10, 'Color', 'white'); 
        title(['Rostro con color: ', color]);
        pause(1);
        hold off;
    end

    % Deformaciones progresivas
    for t = linspace(0, 1, 20)
        deformedImage = I; 
        scaleFactor = 1 + 0.4 * t; % Escala de la deformación

        % ojo 1
        eye1Crop = imcrop(I, eye1);
        resizedEye1 = imresize(eye1Crop, scaleFactor);
        [h, w, ~] = size(resizedEye1);
        xOffset = round((w - eye1(3)) / 2);
        yOffset = round((h - eye1(4)) / 2);
        deformedImage(eye1(2)-yOffset:eye1(2)+h-yOffset-1, eye1(1)-xOffset:eye1(1)+w-xOffset-1, :) = resizedEye1;

        %  ojo 2
        eye2Crop = imcrop(I, eye2);
        resizedEye2 = imresize(eye2Crop, scaleFactor);
        [h, w, ~] = size(resizedEye2);
        xOffset = round((w - eye2(3)) / 2);
        yOffset = round((h - eye2(4)) / 2);
        deformedImage(eye2(2)-yOffset:eye2(2)+h-yOffset-1, eye2(1)-xOffset:eye2(1)+w-xOffset-1, :) = resizedEye2;

        %  nariz
        noseCrop = imcrop(I, nose);
        resizedNose = imresize(noseCrop, scaleFactor);
        [h, w, ~] = size(resizedNose);
        xOffset = round((w - nose(3)) / 2);
        yOffset = round((h - nose(4)) / 2);
        deformedImage(nose(2)-yOffset:nose(2)+h-yOffset-1, nose(1)-xOffset:nose(1)+w-xOffset-1, :) = resizedNose;

        %  boca
        mouthCrop = imcrop(I, mouth);
        resizedMouth = imresize(mouthCrop, scaleFactor);
        [h, w, ~] = size(resizedMouth);
        xOffset = round((w - mouth(3)) / 2);
        yOffset = round((h - mouth(4)) / 2);
        deformedImage(mouth(2)-yOffset:mouth(2)+h-yOffset-1, mouth(1)-xOffset:mouth(1)+w-xOffset-1, :) = resizedMouth;

       
        imshow(deformedImage);
        hold on;
        plot(shape(:,1), shape(:,2), '+', 'MarkerSize', 10, 'Color', 'white'); 
        title(['Paso de deformación: ', num2str(round(t * 20))]);
        pause(0.1);
        hold off;
    end

else
    disp('No se detectó ningún rostro.');
end
