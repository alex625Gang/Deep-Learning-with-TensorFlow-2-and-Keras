	�:M�;@�:M�;@!�:M�;@	z��:�?z��:�?!z��:�?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$�:M�;@K�46�?AX�2ı�:@Y�������?*	�����7t@2F
Iterator::Modelo��ʡ�?!dyB��Q@)O��e��?1�,�t�P@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat-C��6�?!=�А^�/@)�J�4�?1.��d�*@:Preprocessing2U
Iterator::Model::ParallelMapV2	�^)ː?!�Ŕ8G@)	�^)ː?1�Ŕ8G@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice���_vO�?!lǸ��L@)���_vO�?1lǸ��L@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate���S㥛?!�+^�B� @)�HP��?1A"�+@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��ǘ���?!:4���@)��ǘ���?1:4���@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipZd;�O��?!s��^p<@)lxz�,C|?1E��8@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapz6�>W[�?!S�Q�$@)lxz�,C|?1E��8@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 1.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9z��:�?#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	K�46�?K�46�?!K�46�?      ��!       "      ��!       *      ��!       2	X�2ı�:@X�2ı�:@!X�2ı�:@:      ��!       B      ��!       J	�������?�������?!�������?R      ��!       Z	�������?�������?!�������?JCPU_ONLYYz��:�?b 