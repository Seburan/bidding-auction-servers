
сА

BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
Г
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
С
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring Ј
@
StaticRegexFullMatch	
input

output
"
patternstring
L

StringJoin
inputs*N

output"

Nint("
	separatorstring "serve*2.14.02v2.14.0-rc1-21-g4dacf3f368e8аc
y
serving_default_args_0Placeholder*'
_output_shapes
:џџџџџџџџџ
*
dtype0*
shape:џџџџџџџџџ

{
serving_default_args_0_1Placeholder*'
_output_shapes
:џџџџџџџџџ
*
dtype0*
shape:џџџџџџџџџ

{
serving_default_args_0_2Placeholder*'
_output_shapes
:џџџџџџџџџ*
dtype0	*
shape:џџџџџџџџџ
{
serving_default_args_0_3Placeholder*'
_output_shapes
:џџџџџџџџџ*
dtype0	*
shape:џџџџџџџџџ
{
serving_default_args_0_4Placeholder*'
_output_shapes
:џџџџџџџџџ*
dtype0	*
shape:џџџџџџџџџ
{
serving_default_args_0_5Placeholder*'
_output_shapes
:џџџџџџџџџ*
dtype0	*
shape:џџџџџџџџџ
{
serving_default_args_0_6Placeholder*'
_output_shapes
:џџџџџџџџџ*
dtype0	*
shape:џџџџџџџџџ
О
PartitionedCallPartitionedCallserving_default_args_0serving_default_args_0_1serving_default_args_0_2serving_default_args_0_3serving_default_args_0_4serving_default_args_0_5serving_default_args_0_6*
Tin
	2					*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference_signature_wrapper_665

NoOpNoOp

ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*B
value9B7 B1


signatures* 

serving_default* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCallStatefulPartitionedCallsaver_filenameConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *%
f R
__inference__traced_save_693

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__traced_restore_702S
х


!__inference_signature_wrapper_665

args_0
args_0_1
args_0_2	
args_0_3	
args_0_4	
args_0_5	
args_0_6	
identityУ
PartitionedCallPartitionedCallargs_0args_0_1args_0_2args_0_3args_0_4args_0_5args_0_6*
Tin
	2					*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *
fR
__inference_pruned_653`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:џџџџџџџџџ
:џџџџџџџџџ
:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ

 
_user_specified_nameargs_0:QM
'
_output_shapes
:џџџџџџџџџ

"
_user_specified_name
args_0_1:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
args_0_2:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
args_0_3:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
args_0_4:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
args_0_5:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
args_0_6
њ.

__inference_pruned_653

args_0
args_0_1
args_0_2	
args_0_3	
args_0_4	
args_0_5	
args_0_6	
identity_
model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ч
*model/dense/MatMul/ReadVariableOp/resourceConst*
_output_shapes

:
*
dtype0*
valueњBї
"шњО\ЁН$y%ОѓqС>хѓКОвBE>Н>ЬоЮ=оОЪЗОаЪОHњ)Н!=АОЦЈО4дНиНЃ>P=У?ГОD RОЃ9ОЅjв>ТH>рvьНюаPОJon>ЭsО>кЂО@[§;ЕГ>ЧуMО`#cНxJДН№>ЇЙ> ZJ=д`=ќќЩОзгЕ> c>>Є, =Вo>rЬ*>^У­ОаќТМСљ>hP­=@e#НFЌ>1<ДфО№$НNд>T>ЙaiОЙЂЖ>EОИ~u=yЏ>3iМ>ИmНѓaQОvrОЦI>z_>XБ=шы)ОY3>:ЧO><
ЃОЉ_НОР~=уБ>ФьМОрЇОды='>2Т<>єіџНЎњH>цb>е>№oОьЅДНЊ>кi|ОЛКР>я> ы%<iК>Ыg>/ГЊ>ю4%>й4ВОnr~>ту=,Оw`0ОPКНІehОіЈОщB>џ~>&'>ЂE=№P=L@ОA\О]AР>Л-Н&V>!О\АМНаy=ОР$ОqЂК>dз=ЄИЕ=Ъ&О=XГЬОИlЏН:,?ОHH=\MО`mНуц>ЉОрm=2ЪНЋ>Ъ>ЌО@ОCЮ>DО>јQО30>№јН}Б>АЇМh>ј-ІНЉlЯ>S5Ъ>џІУ>Ј >ђt>/а>жш>>BО­Л>жОќєН=Ћ Л> 	=;)/> ѓ/НZОЪ	 >ј]О ?ЋНС>РKНоЧОн7TО$H>§$>>л->`Щ"=R8ОЂq><Y=мЙ=чш= =АеЉО3я>X=њ'>Ў]>>S+[ОKДНЗЊОв>НОњбО№НО.ђA>єЊР=>XъЄНюпКОЖs>ЂJ>­=БЂ>eОАdНж,|>Т?yОњЩ>xСMН$ЂЫ=
L>ЈЇ}НњГ=@ООO>Т!ЏОВR>y
Ф>аxНiОK9>JTО,Цљ=9Р>#Й>јч/= 5;Іl>юX>PІАН2o>]О" >в+О8кОон>ХЃв>зAОњОиaЊОю)>5О l~Н^?n>оЩЁОV6*>RЁ>mCІ>e">`Э)=b_ЁО` |НЬгНлЅЎ>тD_>
+model/dense/BiasAdd/ReadVariableOp/resourceConst*
_output_shapes
:
*
dtype0*=
value4B2
"(                                        Ѕ
,model/dense_1/MatMul/ReadVariableOp/resourceConst*
_output_shapes

:
*
dtype0*A
value8B6
"(.3? J<*<ПкОАКЄ=шѓ4>ЪОo>жы</VчОz
-model/dense_1/BiasAdd/ReadVariableOp/resourceConst*
_output_shapes
:*
dtype0*
valueB*    g
model/concatenate/CastCastargs_0*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџ
k
model/concatenate/Cast_1Castargs_0_1*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџ
e
model/tf.cast/CastCastargs_0_2*

DstT0*

SrcT0	*'
_output_shapes
:џџџџџџџџџy
model/concatenate/Cast_2Castmodel/tf.cast/Cast:y:0*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџg
model/tf.cast_1/CastCastargs_0_3*

DstT0*

SrcT0	*'
_output_shapes
:џџџџџџџџџ{
model/concatenate/Cast_3Castmodel/tf.cast_1/Cast:y:0*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџg
model/tf.cast_2/CastCastargs_0_4*

DstT0*

SrcT0	*'
_output_shapes
:џџџџџџџџџ{
model/concatenate/Cast_4Castmodel/tf.cast_2/Cast:y:0*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџg
model/tf.cast_3/CastCastargs_0_5*

DstT0*

SrcT0	*'
_output_shapes
:џџџџџџџџџ{
model/concatenate/Cast_5Castmodel/tf.cast_3/Cast:y:0*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџg
model/tf.cast_4/CastCastargs_0_6*

DstT0*

SrcT0	*'
_output_shapes
:џџџџџџџџџ{
model/concatenate/Cast_6Castmodel/tf.cast_4/Cast:y:0*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџз
model/concatenate/concatConcatV2model/concatenate/Cast:y:0model/concatenate/Cast_1:y:0model/concatenate/Cast_2:y:0model/concatenate/Cast_3:y:0model/concatenate/Cast_4:y:0model/concatenate/Cast_5:y:0model/concatenate/Cast_6:y:0&model/concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџГ
!model/dense/MatMul/ReadVariableOpIdentity3model/dense/MatMul/ReadVariableOp/resource:output:0*
T0*&
 _has_manual_control_dependencies(*
_output_shapes

:

model/dense/MatMulMatMul!model/concatenate/concat:output:0*model/dense/MatMul/ReadVariableOp:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
Б
"model/dense/BiasAdd/ReadVariableOpIdentity4model/dense/BiasAdd/ReadVariableOp/resource:output:0*
T0*&
 _has_manual_control_dependencies(*
_output_shapes
:

model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0+model/dense/BiasAdd/ReadVariableOp:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
h
model/dense/ReluRelumodel/dense/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
З
#model/dense_1/MatMul/ReadVariableOpIdentity5model/dense_1/MatMul/ReadVariableOp/resource:output:0*
T0*&
 _has_manual_control_dependencies(*
_output_shapes

:

model/dense_1/MatMulMatMulmodel/dense/Relu:activations:0,model/dense_1/MatMul/ReadVariableOp:output:0*
T0*'
_output_shapes
:џџџџџџџџџЕ
$model/dense_1/BiasAdd/ReadVariableOpIdentity6model/dense_1/BiasAdd/ReadVariableOp/resource:output:0*
T0*&
 _has_manual_control_dependencies(*
_output_shapes
:Ё
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0-model/dense_1/BiasAdd/ReadVariableOp:output:0*
T0*'
_output_shapes
:џџџџџџџџџr
model/dense_1/SigmoidSigmoidmodel/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџр
NoOpNoOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 h
IdentityIdentitymodel/dense_1/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:џџџџџџџџџ
:џџџџџџџџџ
:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:- )
'
_output_shapes
:џџџџџџџџџ
:-)
'
_output_shapes
:џџџџџџџџџ
:-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ

E
__inference__traced_restore_702
file_prefix

identity_1
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPHr
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B Ѓ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes
:*
dtypes
2Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 X
IdentityIdentityfile_prefix^NoOp"/device:CPU:0*
T0*
_output_shapes
: J

Identity_1IdentityIdentity:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: :C ?

_output_shapes
: 
%
_user_specified_namefile_prefix

i
__inference__traced_save_693
file_prefix
savev2_const

identity_1ЂMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPHo
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B и
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes
2
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Г
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 7
NoOpNoOp^MergeV2Checkpoints*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:=9

_output_shapes
: 

_user_specified_nameConst"ЇJ
saver_filename:0StatefulPartitionedCall:0StatefulPartitionedCall_18"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*
serving_default
9
args_0/
serving_default_args_0:0џџџџџџџџџ

=
args_0_11
serving_default_args_0_1:0џџџџџџџџџ

=
args_0_21
serving_default_args_0_2:0	џџџџџџџџџ
=
args_0_31
serving_default_args_0_3:0	џџџџџџџџџ
=
args_0_41
serving_default_args_0_4:0	џџџџџџџџџ
=
args_0_51
serving_default_args_0_5:0	џџџџџџџџџ
=
args_0_61
serving_default_args_0_6:0	џџџџџџџџџ4
output_0(
PartitionedCall:0џџџџџџџџџtensorflow/serving/predict:


signaturesB^
__inference_pruned_653args_0args_0_1args_0_2args_0_3args_0_4args_0_5args_0_6z
signatures
,
serving_default"
signature_map
ЭBЪ
!__inference_signature_wrapper_665args_0args_0_1args_0_2args_0_3args_0_4args_0_5args_0_6"р
йВе
FullArgSpec
args 
varargs
 
varkw
 
defaults
 c

kwonlyargsUR
jargs_0

jargs_0_1

jargs_0_2

jargs_0_3

jargs_0_4

jargs_0_5

jargs_0_6
kwonlydefaults
 
annotationsЊ *
 G
__inference_pruned_653-
 "'$
"
tensor_0џџџџџџџџџК
!__inference_signature_wrapper_665мЂи
Ђ 
аЊЬ
*
args_0 
args_0џџџџџџџџџ

.
args_0_1"
args_0_1џџџџџџџџџ

.
args_0_2"
args_0_2џџџџџџџџџ	
.
args_0_3"
args_0_3џџџџџџџџџ	
.
args_0_4"
args_0_4џџџџџџџџџ	
.
args_0_5"
args_0_5џџџџџџџџџ	
.
args_0_6"
args_0_6џџџџџџџџџ	"3Њ0
.
output_0"
output_0џџџџџџџџџ