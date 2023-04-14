folder_path = "Z:/grodriguez/CardiacOCT/data-original/extra-scans-DICOM-3";
file = dicominfo(folder_path);
files = dir(fullfile(folder_path, '*.dcm'));
 
output_table = table(cell(length(files),1), zeros(length(files),1), zeros(length(files),1), 'VariableNames', {'Pullback', 'Shape', 'Spacing'});
 
for i=1:length(files)
     file_path = fullfile(folder_path, files(i).name);
     file = dicominfo(file_path);
     spacing = file.PixelSpacing;
     shape = [file.Rows, file.Columns];
 
     output_table.Pullback{i} = files(i).name;
     output_table.Shape(i) = spacing(1);
     output_table.Spacing(i) = shape(1);
end
 
writetable(output_table, 'spacing.xlsx')