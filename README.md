# Non-Invasive-method-of-Hemoglobin-and-Glucose-level-detection
Hemoglobin concentration is a vital diagnostic parameter. It is essential to
measure and keep track of the Hemoglobin level in the body in order to
diagnose anemia(a condition in which you lack enough healthy red blood cells
to carry adequate oxygen to your body's tissues) and polycythemia (a condition
of blood cancer where more red cells are produced by bone marrow than the
normal level). Since non-invasive hemoglobin measurement methods with high
precision have not been developed yet, only invasive technologies are in
use.This proposed work focuses on measurement techniques of hemoglobin
level and improving the technique of non-invasive methods. Also the work
merges the non invasive method of glucose , heart rate, Spo2 and Blood
Pressure measurement. The non-invasive measurements of above vitals are
mainly based on the light absorption by the red blood cells(hemoglobin) in the
blood and the analysis of resulting Photoplethysmography (PPG) signals. PPG
signal is analyzed by using the modulation ratio which contains the ratios of AC
and DC components from red and IR wavelength light. This proposed method
focuses on establishing the relationship between the hemoglobin concentration
in the blood and modulation ratio and other significant parameters to build a
non-invasive approach.

Oxygen saturation level can be identified by employing a photodetector,
red and near-IR LED’s to measure the light that scatters through blood perfused
tissue. Oxygen is transported in the blood by hemoglobin. Depending on whether 
hemoglobin is bound to oxygen, it absorbs light at different wavelengths.
The datasets will be obtained from the subjects using a hemoglobin level
detector machine invasively along with the PPG signal . The obtained PPG
signals are filtered and processed to extract the first derivative and second
derivative features that have a relationship bounding to the oxygen carrying
hemoglobin.Since the modulation ratio is based on PPG signals as shown in
equation (1), it is clear that there is a relationship between the modulation ratio
(R) and the hemoglobin concentration level.All these features along with the obtained Hb concentrations are used to
train the regression model and to come up with an efficient model to test with
real world data to estimate the hemoglobin concentration non-invasively. The
proposed method
![Glucose prediction accuracy](https://github.com/sudarshanellora/Non-Invasive-method-of-Hemoglobin-and-Glucose-level-detection/assets/115383952/03a45276-8c02-4ec2-b2ea-969541ef0dbe)
![Model testing](https://github.com/sudarshanellora/Non-Invasive-method-of-Hemoglobin-and-Glucose-level-detection/assets/115383952/31231fdd-ca94-49b9-be11-acedfe5798b6)
![Model prediction](https://github.com/sudarshanellora/Non-Invasive-method-of-Hemoglobin-and-Glucose-level-detection/assets/115383952/08d0aed9-607a-4d6e-9065-8303d703d49c)
![Hemoglobin prediction accuracy](https://github.com/sudarshanellora/Non-Invasive-method-of-Hemoglobin-and-Glucose-level-detection/assets/115383952/bc8b380a-fad7-4bb7-8b0b-20c56a86045b)


