% put in data directory
files = dir('*.mat');  % Get all .mat files in the current directory

% Loop directly over the files
for file = files'
    disp(file.name);  % Display the name of each .mat file
    convertTablesToStruct(file.name)
end

function convertTablesToStruct(filepath)
    % Load the .mat file
    matData = load(filepath);
    
    % Recursively convert tables to structs in the loaded data
    matData = convertTableToStructRecursive(matData);
    
    % Optionally, save the modified data back to a new .mat file
    newFileName = strrep(filepath, '.mat', '_converted.mat');
    save(newFileName, '-struct', 'matData');
end

function outData = convertTableToStructRecursive(inData)
    % Initialize the output data structure
    outData = inData;
    
    % Get the field names (if input is a struct)
    if isstruct(inData)
        fields = fieldnames(inData);
        
        for i = 1:numel(fields)
            field = fields{i};
            fieldData = inData.(field);
            
            % If the field is a table, convert it to a struct
            if istable(fieldData)
                outData.(field) = table2struct(fieldData);
            % If the field is a nested struct, recurse
            elseif isstruct(fieldData)
                outData.(field) = convertTableToStructRecursive(fieldData);
            end
        end
    end
end
