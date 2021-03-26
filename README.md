# Eye movements feature extraction for Tobii Desktop eye tracker
All code to process new version data is in eye_extract.py.<br>
Entry point is the function below.<br>
```
extract_save_emotion_eye_fea(xlsx_path_list_for_a_clip,
                             save_path_list_for_a_clip,
                             trigger_list_for_a_clip,
                             window_size,
                             overlap_rate,
                             sample_freq,
                             fea_type,
                             interpolate_type)
```
To remove luminance influences on eye pupil diameters, I process all data associated with a specified video clip at one time.<br>
`xlsx_path_list_for_a_clip` is a list of all xlsx files for a specified video clip.<br>
`save_path_list_for_a_clip` is a list of saving paths for the eye features extracted from each input xlsx file respectively.<br>
`trigger_list_for_a_clip` is a list of trigger pairs to locate the wanted samples in each xlsx file respectively.<br>
`window_size` is the number of seconds of a window to calculate PSD and DE features.<br>
`overlap_rate` is the ratio of overlap size over window size.<br>
`sample_freq` is the sampling frequency of the eye tracker.<br>
`fea_type` can be PSD or DE. <br>
`interpolate_type` is the interpolation method used in `pandas.DataFrame.interpolate`
