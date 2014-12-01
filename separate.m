% Separate each different emotion type.
function [imSetAnger, imSetDisgust, imSetFear, imSetHappy, imSetSad, imSetSurprise, imSetNeutral] = separate(tr_images, tr_label)

anger = find(tr_label == 1);
disgust = find(tr_label == 2);
fear = find(tr_label == 3);
happy = find(tr_label == 4);
sad = find(tr_label == 5);
surprise = find(tr_label == 6);
neutral = find(tr_label == 7);

imSetAnger = tr_images(:,:,anger);
imSetDisgust = tr_images(:,:,disgust);
imSetFear = tr_images(:,:,fear);
imSetHappy = tr_images(:,:,happy);
imSetSad = tr_images(:,:,sad);
imSetSurprise = tr_images(:,:,surprise);
imSetNeutral = tr_images(:,:,neutral);

end