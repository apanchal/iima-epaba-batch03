       �K"	   }_��Abrain.Event:2}̋~��     �Z��	6X1}_��A"��
r
dense_1_inputPlaceholder*
dtype0*(
_output_shapes
:����������*
shape:����������
m
dense_1/random_uniform/shapeConst*
valueB"   �   *
dtype0*
_output_shapes
:
_
dense_1/random_uniform/minConst*
valueB
 *�\1�*
dtype0*
_output_shapes
: 
_
dense_1/random_uniform/maxConst*
valueB
 *�\1=*
dtype0*
_output_shapes
: 
�
$dense_1/random_uniform/RandomUniformRandomUniformdense_1/random_uniform/shape*
T0*
dtype0* 
_output_shapes
:
��*
seed2���*
seed���)
z
dense_1/random_uniform/subSubdense_1/random_uniform/maxdense_1/random_uniform/min*
T0*
_output_shapes
: 
�
dense_1/random_uniform/mulMul$dense_1/random_uniform/RandomUniformdense_1/random_uniform/sub*
T0* 
_output_shapes
:
��
�
dense_1/random_uniformAdddense_1/random_uniform/muldense_1/random_uniform/min*
T0* 
_output_shapes
:
��
�
dense_1/kernelVarHandleOp*
	container *
shape:
��*
dtype0*
_output_shapes
: *
shared_namedense_1/kernel*!
_class
loc:@dense_1/kernel
m
/dense_1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_1/kernel*
_output_shapes
: 
^
dense_1/kernel/AssignAssignVariableOpdense_1/kerneldense_1/random_uniform*
dtype0
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
dtype0* 
_output_shapes
:
��
\
dense_1/ConstConst*
valueB�*    *
dtype0*
_output_shapes	
:�
�
dense_1/biasVarHandleOp*
dtype0*
_output_shapes
: *
shared_namedense_1/bias*
_class
loc:@dense_1/bias*
	container *
shape:�
i
-dense_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_1/bias*
_output_shapes
: 
Q
dense_1/bias/AssignAssignVariableOpdense_1/biasdense_1/Const*
dtype0
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
dtype0*
_output_shapes	
:�
n
dense_1/MatMul/ReadVariableOpReadVariableOpdense_1/kernel*
dtype0* 
_output_shapes
:
��
�
dense_1/MatMulMatMuldense_1_inputdense_1/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b( 
h
dense_1/BiasAdd/ReadVariableOpReadVariableOpdense_1/bias*
dtype0*
_output_shapes	
:�
�
dense_1/BiasAddBiasAdddense_1/MatMuldense_1/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*(
_output_shapes
:����������
X
dense_1/ReluReludense_1/BiasAdd*
T0*(
_output_shapes
:����������
j
batch_normalization_1/ConstConst*
valueB�*  �?*
dtype0*
_output_shapes	
:�
�
batch_normalization_1/gammaVarHandleOp*
shape:�*
dtype0*
_output_shapes
: *,
shared_namebatch_normalization_1/gamma*.
_class$
" loc:@batch_normalization_1/gamma*
	container 
�
<batch_normalization_1/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_1/gamma*
_output_shapes
: 
}
"batch_normalization_1/gamma/AssignAssignVariableOpbatch_normalization_1/gammabatch_normalization_1/Const*
dtype0
�
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
dtype0*
_output_shapes	
:�
l
batch_normalization_1/Const_1Const*
valueB�*    *
dtype0*
_output_shapes	
:�
�
batch_normalization_1/betaVarHandleOp*-
_class#
!loc:@batch_normalization_1/beta*
	container *
shape:�*
dtype0*
_output_shapes
: *+
shared_namebatch_normalization_1/beta
�
;batch_normalization_1/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_1/beta*
_output_shapes
: 
}
!batch_normalization_1/beta/AssignAssignVariableOpbatch_normalization_1/betabatch_normalization_1/Const_1*
dtype0
�
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
dtype0*
_output_shapes	
:�
l
batch_normalization_1/Const_2Const*
valueB�*    *
dtype0*
_output_shapes	
:�
�
!batch_normalization_1/moving_meanVarHandleOp*4
_class*
(&loc:@batch_normalization_1/moving_mean*
	container *
shape:�*
dtype0*
_output_shapes
: *2
shared_name#!batch_normalization_1/moving_mean
�
Bbatch_normalization_1/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp!batch_normalization_1/moving_mean*
_output_shapes
: 
�
(batch_normalization_1/moving_mean/AssignAssignVariableOp!batch_normalization_1/moving_meanbatch_normalization_1/Const_2*
dtype0
�
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
dtype0*
_output_shapes	
:�
l
batch_normalization_1/Const_3Const*
dtype0*
_output_shapes	
:�*
valueB�*  �?
�
%batch_normalization_1/moving_varianceVarHandleOp*6
shared_name'%batch_normalization_1/moving_variance*8
_class.
,*loc:@batch_normalization_1/moving_variance*
	container *
shape:�*
dtype0*
_output_shapes
: 
�
Fbatch_normalization_1/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp%batch_normalization_1/moving_variance*
_output_shapes
: 
�
,batch_normalization_1/moving_variance/AssignAssignVariableOp%batch_normalization_1/moving_variancebatch_normalization_1/Const_3*
dtype0
�
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
dtype0*
_output_shapes	
:�
~
4batch_normalization_1/moments/mean/reduction_indicesConst*
valueB: *
dtype0*
_output_shapes
:
�
"batch_normalization_1/moments/meanMeandense_1/Relu4batch_normalization_1/moments/mean/reduction_indices*
T0*
_output_shapes
:	�*
	keep_dims(*

Tidx0
�
*batch_normalization_1/moments/StopGradientStopGradient"batch_normalization_1/moments/mean*
T0*
_output_shapes
:	�
�
/batch_normalization_1/moments/SquaredDifferenceSquaredDifferencedense_1/Relu*batch_normalization_1/moments/StopGradient*
T0*(
_output_shapes
:����������
�
8batch_normalization_1/moments/variance/reduction_indicesConst*
valueB: *
dtype0*
_output_shapes
:
�
&batch_normalization_1/moments/varianceMean/batch_normalization_1/moments/SquaredDifference8batch_normalization_1/moments/variance/reduction_indices*
T0*
_output_shapes
:	�*
	keep_dims(*

Tidx0
�
%batch_normalization_1/moments/SqueezeSqueeze"batch_normalization_1/moments/mean*
squeeze_dims
 *
T0*
_output_shapes	
:�
�
'batch_normalization_1/moments/Squeeze_1Squeeze&batch_normalization_1/moments/variance*
squeeze_dims
 *
T0*
_output_shapes	
:�
j
%batch_normalization_1/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *o�:
�
#batch_normalization_1/batchnorm/addAddV2'batch_normalization_1/moments/Squeeze_1%batch_normalization_1/batchnorm/add/y*
T0*
_output_shapes	
:�
y
%batch_normalization_1/batchnorm/RsqrtRsqrt#batch_normalization_1/batchnorm/add*
T0*
_output_shapes	
:�
�
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
dtype0*
_output_shapes	
:�
�
#batch_normalization_1/batchnorm/mulMul%batch_normalization_1/batchnorm/Rsqrt2batch_normalization_1/batchnorm/mul/ReadVariableOp*
T0*
_output_shapes	
:�
�
%batch_normalization_1/batchnorm/mul_1Muldense_1/Relu#batch_normalization_1/batchnorm/mul*
T0*(
_output_shapes
:����������
�
%batch_normalization_1/batchnorm/mul_2Mul%batch_normalization_1/moments/Squeeze#batch_normalization_1/batchnorm/mul*
T0*
_output_shapes	
:�
�
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
dtype0*
_output_shapes	
:�
�
#batch_normalization_1/batchnorm/subSub.batch_normalization_1/batchnorm/ReadVariableOp%batch_normalization_1/batchnorm/mul_2*
_output_shapes	
:�*
T0
�
%batch_normalization_1/batchnorm/add_1AddV2%batch_normalization_1/batchnorm/mul_1#batch_normalization_1/batchnorm/sub*
T0*(
_output_shapes
:����������
g
batch_normalization_1/ShapeShapedense_1/Relu*
T0*
out_type0*
_output_shapes
:
s
)batch_normalization_1/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
u
+batch_normalization_1/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
u
+batch_normalization_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
#batch_normalization_1/strided_sliceStridedSlicebatch_normalization_1/Shape)batch_normalization_1/strided_slice/stack+batch_normalization_1/strided_slice/stack_1+batch_normalization_1/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
�
!batch_normalization_1/Rank/packedPack#batch_normalization_1/strided_slice*
T0*

axis *
N*
_output_shapes
:
\
batch_normalization_1/RankConst*
dtype0*
_output_shapes
: *
value	B :
c
!batch_normalization_1/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
c
!batch_normalization_1/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
�
batch_normalization_1/rangeRange!batch_normalization_1/range/startbatch_normalization_1/Rank!batch_normalization_1/range/delta*
_output_shapes
:*

Tidx0
�
 batch_normalization_1/Prod/inputPack#batch_normalization_1/strided_slice*
T0*

axis *
N*
_output_shapes
:
�
batch_normalization_1/ProdProd batch_normalization_1/Prod/inputbatch_normalization_1/range*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
~
batch_normalization_1/CastCastbatch_normalization_1/Prod*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
`
batch_normalization_1/sub/yConst*
dtype0*
_output_shapes
: *
valueB
 *� �?
z
batch_normalization_1/subSubbatch_normalization_1/Castbatch_normalization_1/sub/y*
T0*
_output_shapes
: 
�
batch_normalization_1/truedivRealDivbatch_normalization_1/Castbatch_normalization_1/sub*
T0*
_output_shapes
: 
�
batch_normalization_1/mulMul'batch_normalization_1/moments/Squeeze_1batch_normalization_1/truediv*
T0*
_output_shapes	
:�
�
batch_normalization_1/Const_4Const*
valueB
 *
�#<*4
_class*
(&loc:@batch_normalization_1/moving_mean*
dtype0*
_output_shapes
: 
�
$batch_normalization_1/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
dtype0*
_output_shapes	
:�
�
batch_normalization_1/sub_1Sub$batch_normalization_1/ReadVariableOp%batch_normalization_1/moments/Squeeze*
_output_shapes	
:�*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean
�
batch_normalization_1/mul_1Mulbatch_normalization_1/sub_1batch_normalization_1/Const_4*
_output_shapes	
:�*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean
�
)batch_normalization_1/AssignSubVariableOpAssignSubVariableOp!batch_normalization_1/moving_meanbatch_normalization_1/mul_1*
dtype0*4
_class*
(&loc:@batch_normalization_1/moving_mean
�
&batch_normalization_1/ReadVariableOp_1ReadVariableOp!batch_normalization_1/moving_mean*^batch_normalization_1/AssignSubVariableOp*4
_class*
(&loc:@batch_normalization_1/moving_mean*
dtype0*
_output_shapes	
:�
�
batch_normalization_1/Const_5Const*
dtype0*
_output_shapes
: *
valueB
 *
�#<*8
_class.
,*loc:@batch_normalization_1/moving_variance
�
&batch_normalization_1/ReadVariableOp_2ReadVariableOp%batch_normalization_1/moving_variance*
dtype0*
_output_shapes	
:�
�
batch_normalization_1/sub_2Sub&batch_normalization_1/ReadVariableOp_2batch_normalization_1/mul*
_output_shapes	
:�*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance
�
batch_normalization_1/mul_2Mulbatch_normalization_1/sub_2batch_normalization_1/Const_5*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes	
:�
�
+batch_normalization_1/AssignSubVariableOp_1AssignSubVariableOp%batch_normalization_1/moving_variancebatch_normalization_1/mul_2*8
_class.
,*loc:@batch_normalization_1/moving_variance*
dtype0
�
&batch_normalization_1/ReadVariableOp_3ReadVariableOp%batch_normalization_1/moving_variance,^batch_normalization_1/AssignSubVariableOp_1*8
_class.
,*loc:@batch_normalization_1/moving_variance*
dtype0*
_output_shapes	
:�
\
keras_learning_phase/inputConst*
dtype0
*
_output_shapes
: *
value	B
 Z 
|
keras_learning_phasePlaceholderWithDefaultkeras_learning_phase/input*
dtype0
*
_output_shapes
: *
shape: 
z
!batch_normalization_1/cond/SwitchSwitchkeras_learning_phasekeras_learning_phase*
T0
*
_output_shapes
: : 
u
#batch_normalization_1/cond/switch_tIdentity#batch_normalization_1/cond/Switch:1*
T0
*
_output_shapes
: 
s
#batch_normalization_1/cond/switch_fIdentity!batch_normalization_1/cond/Switch*
T0
*
_output_shapes
: 
e
"batch_normalization_1/cond/pred_idIdentitykeras_learning_phase*
_output_shapes
: *
T0

�
#batch_normalization_1/cond/Switch_1Switch%batch_normalization_1/batchnorm/add_1"batch_normalization_1/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1*<
_output_shapes*
(:����������:����������
�
3batch_normalization_1/cond/batchnorm/ReadVariableOpReadVariableOp:batch_normalization_1/cond/batchnorm/ReadVariableOp/Switch*
dtype0*
_output_shapes	
:�
�
:batch_normalization_1/cond/batchnorm/ReadVariableOp/SwitchSwitch%batch_normalization_1/moving_variance"batch_normalization_1/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes
: : 
�
*batch_normalization_1/cond/batchnorm/add/yConst$^batch_normalization_1/cond/switch_f*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
(batch_normalization_1/cond/batchnorm/addAddV23batch_normalization_1/cond/batchnorm/ReadVariableOp*batch_normalization_1/cond/batchnorm/add/y*
T0*
_output_shapes	
:�
�
*batch_normalization_1/cond/batchnorm/RsqrtRsqrt(batch_normalization_1/cond/batchnorm/add*
T0*
_output_shapes	
:�
�
7batch_normalization_1/cond/batchnorm/mul/ReadVariableOpReadVariableOp>batch_normalization_1/cond/batchnorm/mul/ReadVariableOp/Switch*
dtype0*
_output_shapes	
:�
�
>batch_normalization_1/cond/batchnorm/mul/ReadVariableOp/SwitchSwitchbatch_normalization_1/gamma"batch_normalization_1/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_1/gamma*
_output_shapes
: : 
�
(batch_normalization_1/cond/batchnorm/mulMul*batch_normalization_1/cond/batchnorm/Rsqrt7batch_normalization_1/cond/batchnorm/mul/ReadVariableOp*
T0*
_output_shapes	
:�
�
*batch_normalization_1/cond/batchnorm/mul_1Mul1batch_normalization_1/cond/batchnorm/mul_1/Switch(batch_normalization_1/cond/batchnorm/mul*(
_output_shapes
:����������*
T0
�
1batch_normalization_1/cond/batchnorm/mul_1/SwitchSwitchdense_1/Relu"batch_normalization_1/cond/pred_id*<
_output_shapes*
(:����������:����������*
T0*
_class
loc:@dense_1/Relu
�
5batch_normalization_1/cond/batchnorm/ReadVariableOp_1ReadVariableOp<batch_normalization_1/cond/batchnorm/ReadVariableOp_1/Switch*
dtype0*
_output_shapes	
:�
�
<batch_normalization_1/cond/batchnorm/ReadVariableOp_1/SwitchSwitch!batch_normalization_1/moving_mean"batch_normalization_1/cond/pred_id*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes
: : 
�
*batch_normalization_1/cond/batchnorm/mul_2Mul5batch_normalization_1/cond/batchnorm/ReadVariableOp_1(batch_normalization_1/cond/batchnorm/mul*
_output_shapes	
:�*
T0
�
5batch_normalization_1/cond/batchnorm/ReadVariableOp_2ReadVariableOp<batch_normalization_1/cond/batchnorm/ReadVariableOp_2/Switch*
dtype0*
_output_shapes	
:�
�
<batch_normalization_1/cond/batchnorm/ReadVariableOp_2/SwitchSwitchbatch_normalization_1/beta"batch_normalization_1/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_1/beta*
_output_shapes
: : 
�
(batch_normalization_1/cond/batchnorm/subSub5batch_normalization_1/cond/batchnorm/ReadVariableOp_2*batch_normalization_1/cond/batchnorm/mul_2*
T0*
_output_shapes	
:�
�
*batch_normalization_1/cond/batchnorm/add_1AddV2*batch_normalization_1/cond/batchnorm/mul_1(batch_normalization_1/cond/batchnorm/sub*
T0*(
_output_shapes
:����������
�
 batch_normalization_1/cond/MergeMerge*batch_normalization_1/cond/batchnorm/add_1%batch_normalization_1/cond/Switch_1:1*
T0*
N**
_output_shapes
:����������: 
m
dense_2/random_uniform/shapeConst*
valueB"�   �   *
dtype0*
_output_shapes
:
_
dense_2/random_uniform/minConst*
valueB
 *q��*
dtype0*
_output_shapes
: 
_
dense_2/random_uniform/maxConst*
valueB
 *q�>*
dtype0*
_output_shapes
: 
�
$dense_2/random_uniform/RandomUniformRandomUniformdense_2/random_uniform/shape*
T0*
dtype0* 
_output_shapes
:
��*
seed2ئ�*
seed���)
z
dense_2/random_uniform/subSubdense_2/random_uniform/maxdense_2/random_uniform/min*
T0*
_output_shapes
: 
�
dense_2/random_uniform/mulMul$dense_2/random_uniform/RandomUniformdense_2/random_uniform/sub*
T0* 
_output_shapes
:
��
�
dense_2/random_uniformAdddense_2/random_uniform/muldense_2/random_uniform/min*
T0* 
_output_shapes
:
��
�
dense_2/kernelVarHandleOp*
shared_namedense_2/kernel*!
_class
loc:@dense_2/kernel*
	container *
shape:
��*
dtype0*
_output_shapes
: 
m
/dense_2/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_2/kernel*
_output_shapes
: 
^
dense_2/kernel/AssignAssignVariableOpdense_2/kerneldense_2/random_uniform*
dtype0
s
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
dtype0* 
_output_shapes
:
��
\
dense_2/ConstConst*
valueB�*    *
dtype0*
_output_shapes	
:�
�
dense_2/biasVarHandleOp*
shared_namedense_2/bias*
_class
loc:@dense_2/bias*
	container *
shape:�*
dtype0*
_output_shapes
: 
i
-dense_2/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_2/bias*
_output_shapes
: 
Q
dense_2/bias/AssignAssignVariableOpdense_2/biasdense_2/Const*
dtype0
j
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
dtype0*
_output_shapes	
:�
n
dense_2/MatMul/ReadVariableOpReadVariableOpdense_2/kernel*
dtype0* 
_output_shapes
:
��
�
dense_2/MatMulMatMul batch_normalization_1/cond/Mergedense_2/MatMul/ReadVariableOp*
transpose_b( *
T0*(
_output_shapes
:����������*
transpose_a( 
h
dense_2/BiasAdd/ReadVariableOpReadVariableOpdense_2/bias*
dtype0*
_output_shapes	
:�
�
dense_2/BiasAddBiasAdddense_2/MatMuldense_2/BiasAdd/ReadVariableOp*
data_formatNHWC*(
_output_shapes
:����������*
T0
X
dense_2/ReluReludense_2/BiasAdd*
T0*(
_output_shapes
:����������
j
batch_normalization_2/ConstConst*
dtype0*
_output_shapes	
:�*
valueB�*  �?
�
batch_normalization_2/gammaVarHandleOp*
shape:�*
dtype0*
_output_shapes
: *,
shared_namebatch_normalization_2/gamma*.
_class$
" loc:@batch_normalization_2/gamma*
	container 
�
<batch_normalization_2/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_2/gamma*
_output_shapes
: 
}
"batch_normalization_2/gamma/AssignAssignVariableOpbatch_normalization_2/gammabatch_normalization_2/Const*
dtype0
�
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
dtype0*
_output_shapes	
:�
l
batch_normalization_2/Const_1Const*
valueB�*    *
dtype0*
_output_shapes	
:�
�
batch_normalization_2/betaVarHandleOp*+
shared_namebatch_normalization_2/beta*-
_class#
!loc:@batch_normalization_2/beta*
	container *
shape:�*
dtype0*
_output_shapes
: 
�
;batch_normalization_2/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_2/beta*
_output_shapes
: 
}
!batch_normalization_2/beta/AssignAssignVariableOpbatch_normalization_2/betabatch_normalization_2/Const_1*
dtype0
�
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
dtype0*
_output_shapes	
:�
l
batch_normalization_2/Const_2Const*
valueB�*    *
dtype0*
_output_shapes	
:�
�
!batch_normalization_2/moving_meanVarHandleOp*
dtype0*
_output_shapes
: *2
shared_name#!batch_normalization_2/moving_mean*4
_class*
(&loc:@batch_normalization_2/moving_mean*
	container *
shape:�
�
Bbatch_normalization_2/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp!batch_normalization_2/moving_mean*
_output_shapes
: 
�
(batch_normalization_2/moving_mean/AssignAssignVariableOp!batch_normalization_2/moving_meanbatch_normalization_2/Const_2*
dtype0
�
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
dtype0*
_output_shapes	
:�
l
batch_normalization_2/Const_3Const*
valueB�*  �?*
dtype0*
_output_shapes	
:�
�
%batch_normalization_2/moving_varianceVarHandleOp*
dtype0*
_output_shapes
: *6
shared_name'%batch_normalization_2/moving_variance*8
_class.
,*loc:@batch_normalization_2/moving_variance*
	container *
shape:�
�
Fbatch_normalization_2/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp%batch_normalization_2/moving_variance*
_output_shapes
: 
�
,batch_normalization_2/moving_variance/AssignAssignVariableOp%batch_normalization_2/moving_variancebatch_normalization_2/Const_3*
dtype0
�
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
dtype0*
_output_shapes	
:�
~
4batch_normalization_2/moments/mean/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB: 
�
"batch_normalization_2/moments/meanMeandense_2/Relu4batch_normalization_2/moments/mean/reduction_indices*
T0*
_output_shapes
:	�*
	keep_dims(*

Tidx0
�
*batch_normalization_2/moments/StopGradientStopGradient"batch_normalization_2/moments/mean*
T0*
_output_shapes
:	�
�
/batch_normalization_2/moments/SquaredDifferenceSquaredDifferencedense_2/Relu*batch_normalization_2/moments/StopGradient*
T0*(
_output_shapes
:����������
�
8batch_normalization_2/moments/variance/reduction_indicesConst*
valueB: *
dtype0*
_output_shapes
:
�
&batch_normalization_2/moments/varianceMean/batch_normalization_2/moments/SquaredDifference8batch_normalization_2/moments/variance/reduction_indices*
	keep_dims(*

Tidx0*
T0*
_output_shapes
:	�
�
%batch_normalization_2/moments/SqueezeSqueeze"batch_normalization_2/moments/mean*
_output_shapes	
:�*
squeeze_dims
 *
T0
�
'batch_normalization_2/moments/Squeeze_1Squeeze&batch_normalization_2/moments/variance*
_output_shapes	
:�*
squeeze_dims
 *
T0
j
%batch_normalization_2/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
#batch_normalization_2/batchnorm/addAddV2'batch_normalization_2/moments/Squeeze_1%batch_normalization_2/batchnorm/add/y*
T0*
_output_shapes	
:�
y
%batch_normalization_2/batchnorm/RsqrtRsqrt#batch_normalization_2/batchnorm/add*
T0*
_output_shapes	
:�
�
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
dtype0*
_output_shapes	
:�
�
#batch_normalization_2/batchnorm/mulMul%batch_normalization_2/batchnorm/Rsqrt2batch_normalization_2/batchnorm/mul/ReadVariableOp*
T0*
_output_shapes	
:�
�
%batch_normalization_2/batchnorm/mul_1Muldense_2/Relu#batch_normalization_2/batchnorm/mul*
T0*(
_output_shapes
:����������
�
%batch_normalization_2/batchnorm/mul_2Mul%batch_normalization_2/moments/Squeeze#batch_normalization_2/batchnorm/mul*
T0*
_output_shapes	
:�
�
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
dtype0*
_output_shapes	
:�
�
#batch_normalization_2/batchnorm/subSub.batch_normalization_2/batchnorm/ReadVariableOp%batch_normalization_2/batchnorm/mul_2*
_output_shapes	
:�*
T0
�
%batch_normalization_2/batchnorm/add_1AddV2%batch_normalization_2/batchnorm/mul_1#batch_normalization_2/batchnorm/sub*
T0*(
_output_shapes
:����������
g
batch_normalization_2/ShapeShapedense_2/Relu*
T0*
out_type0*
_output_shapes
:
s
)batch_normalization_2/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
u
+batch_normalization_2/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
u
+batch_normalization_2/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
#batch_normalization_2/strided_sliceStridedSlicebatch_normalization_2/Shape)batch_normalization_2/strided_slice/stack+batch_normalization_2/strided_slice/stack_1+batch_normalization_2/strided_slice/stack_2*
end_mask *
_output_shapes
: *
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask 
�
!batch_normalization_2/Rank/packedPack#batch_normalization_2/strided_slice*
T0*

axis *
N*
_output_shapes
:
\
batch_normalization_2/RankConst*
value	B :*
dtype0*
_output_shapes
: 
c
!batch_normalization_2/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
c
!batch_normalization_2/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
batch_normalization_2/rangeRange!batch_normalization_2/range/startbatch_normalization_2/Rank!batch_normalization_2/range/delta*

Tidx0*
_output_shapes
:
�
 batch_normalization_2/Prod/inputPack#batch_normalization_2/strided_slice*
N*
_output_shapes
:*
T0*

axis 
�
batch_normalization_2/ProdProd batch_normalization_2/Prod/inputbatch_normalization_2/range*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
~
batch_normalization_2/CastCastbatch_normalization_2/Prod*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
`
batch_normalization_2/sub/yConst*
valueB
 *� �?*
dtype0*
_output_shapes
: 
z
batch_normalization_2/subSubbatch_normalization_2/Castbatch_normalization_2/sub/y*
T0*
_output_shapes
: 
�
batch_normalization_2/truedivRealDivbatch_normalization_2/Castbatch_normalization_2/sub*
T0*
_output_shapes
: 
�
batch_normalization_2/mulMul'batch_normalization_2/moments/Squeeze_1batch_normalization_2/truediv*
T0*
_output_shapes	
:�
�
batch_normalization_2/Const_4Const*
dtype0*
_output_shapes
: *
valueB
 *
�#<*4
_class*
(&loc:@batch_normalization_2/moving_mean
�
$batch_normalization_2/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
dtype0*
_output_shapes	
:�
�
batch_normalization_2/sub_1Sub$batch_normalization_2/ReadVariableOp%batch_normalization_2/moments/Squeeze*
_output_shapes	
:�*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean
�
batch_normalization_2/mul_1Mulbatch_normalization_2/sub_1batch_normalization_2/Const_4*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean*
_output_shapes	
:�
�
)batch_normalization_2/AssignSubVariableOpAssignSubVariableOp!batch_normalization_2/moving_meanbatch_normalization_2/mul_1*4
_class*
(&loc:@batch_normalization_2/moving_mean*
dtype0
�
&batch_normalization_2/ReadVariableOp_1ReadVariableOp!batch_normalization_2/moving_mean*^batch_normalization_2/AssignSubVariableOp*
dtype0*
_output_shapes	
:�*4
_class*
(&loc:@batch_normalization_2/moving_mean
�
batch_normalization_2/Const_5Const*
dtype0*
_output_shapes
: *
valueB
 *
�#<*8
_class.
,*loc:@batch_normalization_2/moving_variance
�
&batch_normalization_2/ReadVariableOp_2ReadVariableOp%batch_normalization_2/moving_variance*
dtype0*
_output_shapes	
:�
�
batch_normalization_2/sub_2Sub&batch_normalization_2/ReadVariableOp_2batch_normalization_2/mul*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes	
:�
�
batch_normalization_2/mul_2Mulbatch_normalization_2/sub_2batch_normalization_2/Const_5*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes	
:�
�
+batch_normalization_2/AssignSubVariableOp_1AssignSubVariableOp%batch_normalization_2/moving_variancebatch_normalization_2/mul_2*8
_class.
,*loc:@batch_normalization_2/moving_variance*
dtype0
�
&batch_normalization_2/ReadVariableOp_3ReadVariableOp%batch_normalization_2/moving_variance,^batch_normalization_2/AssignSubVariableOp_1*
dtype0*
_output_shapes	
:�*8
_class.
,*loc:@batch_normalization_2/moving_variance
z
!batch_normalization_2/cond/SwitchSwitchkeras_learning_phasekeras_learning_phase*
_output_shapes
: : *
T0

u
#batch_normalization_2/cond/switch_tIdentity#batch_normalization_2/cond/Switch:1*
T0
*
_output_shapes
: 
s
#batch_normalization_2/cond/switch_fIdentity!batch_normalization_2/cond/Switch*
T0
*
_output_shapes
: 
e
"batch_normalization_2/cond/pred_idIdentitykeras_learning_phase*
T0
*
_output_shapes
: 
�
#batch_normalization_2/cond/Switch_1Switch%batch_normalization_2/batchnorm/add_1"batch_normalization_2/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/add_1*<
_output_shapes*
(:����������:����������
�
3batch_normalization_2/cond/batchnorm/ReadVariableOpReadVariableOp:batch_normalization_2/cond/batchnorm/ReadVariableOp/Switch*
dtype0*
_output_shapes	
:�
�
:batch_normalization_2/cond/batchnorm/ReadVariableOp/SwitchSwitch%batch_normalization_2/moving_variance"batch_normalization_2/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes
: : 
�
*batch_normalization_2/cond/batchnorm/add/yConst$^batch_normalization_2/cond/switch_f*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
(batch_normalization_2/cond/batchnorm/addAddV23batch_normalization_2/cond/batchnorm/ReadVariableOp*batch_normalization_2/cond/batchnorm/add/y*
T0*
_output_shapes	
:�
�
*batch_normalization_2/cond/batchnorm/RsqrtRsqrt(batch_normalization_2/cond/batchnorm/add*
_output_shapes	
:�*
T0
�
7batch_normalization_2/cond/batchnorm/mul/ReadVariableOpReadVariableOp>batch_normalization_2/cond/batchnorm/mul/ReadVariableOp/Switch*
dtype0*
_output_shapes	
:�
�
>batch_normalization_2/cond/batchnorm/mul/ReadVariableOp/SwitchSwitchbatch_normalization_2/gamma"batch_normalization_2/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_2/gamma*
_output_shapes
: : 
�
(batch_normalization_2/cond/batchnorm/mulMul*batch_normalization_2/cond/batchnorm/Rsqrt7batch_normalization_2/cond/batchnorm/mul/ReadVariableOp*
T0*
_output_shapes	
:�
�
*batch_normalization_2/cond/batchnorm/mul_1Mul1batch_normalization_2/cond/batchnorm/mul_1/Switch(batch_normalization_2/cond/batchnorm/mul*(
_output_shapes
:����������*
T0
�
1batch_normalization_2/cond/batchnorm/mul_1/SwitchSwitchdense_2/Relu"batch_normalization_2/cond/pred_id*
T0*
_class
loc:@dense_2/Relu*<
_output_shapes*
(:����������:����������
�
5batch_normalization_2/cond/batchnorm/ReadVariableOp_1ReadVariableOp<batch_normalization_2/cond/batchnorm/ReadVariableOp_1/Switch*
dtype0*
_output_shapes	
:�
�
<batch_normalization_2/cond/batchnorm/ReadVariableOp_1/SwitchSwitch!batch_normalization_2/moving_mean"batch_normalization_2/cond/pred_id*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean*
_output_shapes
: : 
�
*batch_normalization_2/cond/batchnorm/mul_2Mul5batch_normalization_2/cond/batchnorm/ReadVariableOp_1(batch_normalization_2/cond/batchnorm/mul*
T0*
_output_shapes	
:�
�
5batch_normalization_2/cond/batchnorm/ReadVariableOp_2ReadVariableOp<batch_normalization_2/cond/batchnorm/ReadVariableOp_2/Switch*
dtype0*
_output_shapes	
:�
�
<batch_normalization_2/cond/batchnorm/ReadVariableOp_2/SwitchSwitchbatch_normalization_2/beta"batch_normalization_2/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_2/beta*
_output_shapes
: : 
�
(batch_normalization_2/cond/batchnorm/subSub5batch_normalization_2/cond/batchnorm/ReadVariableOp_2*batch_normalization_2/cond/batchnorm/mul_2*
T0*
_output_shapes	
:�
�
*batch_normalization_2/cond/batchnorm/add_1AddV2*batch_normalization_2/cond/batchnorm/mul_1(batch_normalization_2/cond/batchnorm/sub*
T0*(
_output_shapes
:����������
�
 batch_normalization_2/cond/MergeMerge*batch_normalization_2/cond/batchnorm/add_1%batch_normalization_2/cond/Switch_1:1*
T0*
N**
_output_shapes
:����������: 
n
dropout_1/cond/SwitchSwitchkeras_learning_phasekeras_learning_phase*
T0
*
_output_shapes
: : 
]
dropout_1/cond/switch_tIdentitydropout_1/cond/Switch:1*
T0
*
_output_shapes
: 
[
dropout_1/cond/switch_fIdentitydropout_1/cond/Switch*
_output_shapes
: *
T0

Y
dropout_1/cond/pred_idIdentitykeras_learning_phase*
_output_shapes
: *
T0

z
dropout_1/cond/dropout/rateConst^dropout_1/cond/switch_t*
valueB
 *   ?*
dtype0*
_output_shapes
: 
�
dropout_1/cond/dropout/ShapeShape%dropout_1/cond/dropout/Shape/Switch:1*
_output_shapes
:*
T0*
out_type0
�
#dropout_1/cond/dropout/Shape/SwitchSwitch batch_normalization_2/cond/Mergedropout_1/cond/pred_id*<
_output_shapes*
(:����������:����������*
T0*3
_class)
'%loc:@batch_normalization_2/cond/Merge
�
)dropout_1/cond/dropout/random_uniform/minConst^dropout_1/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *    
�
)dropout_1/cond/dropout/random_uniform/maxConst^dropout_1/cond/switch_t*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
3dropout_1/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_1/cond/dropout/Shape*
seed���)*
T0*
dtype0*(
_output_shapes
:����������*
seed2���
�
)dropout_1/cond/dropout/random_uniform/subSub)dropout_1/cond/dropout/random_uniform/max)dropout_1/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
�
)dropout_1/cond/dropout/random_uniform/mulMul3dropout_1/cond/dropout/random_uniform/RandomUniform)dropout_1/cond/dropout/random_uniform/sub*
T0*(
_output_shapes
:����������
�
%dropout_1/cond/dropout/random_uniformAdd)dropout_1/cond/dropout/random_uniform/mul)dropout_1/cond/dropout/random_uniform/min*(
_output_shapes
:����������*
T0
{
dropout_1/cond/dropout/sub/xConst^dropout_1/cond/switch_t*
valueB
 *  �?*
dtype0*
_output_shapes
: 
}
dropout_1/cond/dropout/subSubdropout_1/cond/dropout/sub/xdropout_1/cond/dropout/rate*
T0*
_output_shapes
: 

 dropout_1/cond/dropout/truediv/xConst^dropout_1/cond/switch_t*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
dropout_1/cond/dropout/truedivRealDiv dropout_1/cond/dropout/truediv/xdropout_1/cond/dropout/sub*
T0*
_output_shapes
: 
�
#dropout_1/cond/dropout/GreaterEqualGreaterEqual%dropout_1/cond/dropout/random_uniformdropout_1/cond/dropout/rate*
T0*(
_output_shapes
:����������
�
dropout_1/cond/dropout/mulMul%dropout_1/cond/dropout/Shape/Switch:1dropout_1/cond/dropout/truediv*
T0*(
_output_shapes
:����������
�
dropout_1/cond/dropout/CastCast#dropout_1/cond/dropout/GreaterEqual*

SrcT0
*
Truncate( *(
_output_shapes
:����������*

DstT0
�
dropout_1/cond/dropout/mul_1Muldropout_1/cond/dropout/muldropout_1/cond/dropout/Cast*
T0*(
_output_shapes
:����������
�
dropout_1/cond/Switch_1Switch batch_normalization_2/cond/Mergedropout_1/cond/pred_id*<
_output_shapes*
(:����������:����������*
T0*3
_class)
'%loc:@batch_normalization_2/cond/Merge
�
dropout_1/cond/MergeMergedropout_1/cond/Switch_1dropout_1/cond/dropout/mul_1*
T0*
N**
_output_shapes
:����������: 
m
dense_3/random_uniform/shapeConst*
valueB"�   �   *
dtype0*
_output_shapes
:
_
dense_3/random_uniform/minConst*
valueB
 *q��*
dtype0*
_output_shapes
: 
_
dense_3/random_uniform/maxConst*
valueB
 *q�>*
dtype0*
_output_shapes
: 
�
$dense_3/random_uniform/RandomUniformRandomUniformdense_3/random_uniform/shape*
seed���)*
T0*
dtype0* 
_output_shapes
:
��*
seed2�֌
z
dense_3/random_uniform/subSubdense_3/random_uniform/maxdense_3/random_uniform/min*
T0*
_output_shapes
: 
�
dense_3/random_uniform/mulMul$dense_3/random_uniform/RandomUniformdense_3/random_uniform/sub*
T0* 
_output_shapes
:
��
�
dense_3/random_uniformAdddense_3/random_uniform/muldense_3/random_uniform/min* 
_output_shapes
:
��*
T0
�
dense_3/kernelVarHandleOp*
	container *
shape:
��*
dtype0*
_output_shapes
: *
shared_namedense_3/kernel*!
_class
loc:@dense_3/kernel
m
/dense_3/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_3/kernel*
_output_shapes
: 
^
dense_3/kernel/AssignAssignVariableOpdense_3/kerneldense_3/random_uniform*
dtype0
s
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
dtype0* 
_output_shapes
:
��
\
dense_3/ConstConst*
dtype0*
_output_shapes	
:�*
valueB�*    
�
dense_3/biasVarHandleOp*
shared_namedense_3/bias*
_class
loc:@dense_3/bias*
	container *
shape:�*
dtype0*
_output_shapes
: 
i
-dense_3/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_3/bias*
_output_shapes
: 
Q
dense_3/bias/AssignAssignVariableOpdense_3/biasdense_3/Const*
dtype0
j
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
dtype0*
_output_shapes	
:�
n
dense_3/MatMul/ReadVariableOpReadVariableOpdense_3/kernel*
dtype0* 
_output_shapes
:
��
�
dense_3/MatMulMatMuldropout_1/cond/Mergedense_3/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b( 
h
dense_3/BiasAdd/ReadVariableOpReadVariableOpdense_3/bias*
dtype0*
_output_shapes	
:�
�
dense_3/BiasAddBiasAdddense_3/MatMuldense_3/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*(
_output_shapes
:����������
X
dense_3/ReluReludense_3/BiasAdd*
T0*(
_output_shapes
:����������
j
batch_normalization_3/ConstConst*
valueB�*  �?*
dtype0*
_output_shapes	
:�
�
batch_normalization_3/gammaVarHandleOp*
dtype0*
_output_shapes
: *,
shared_namebatch_normalization_3/gamma*.
_class$
" loc:@batch_normalization_3/gamma*
	container *
shape:�
�
<batch_normalization_3/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_3/gamma*
_output_shapes
: 
}
"batch_normalization_3/gamma/AssignAssignVariableOpbatch_normalization_3/gammabatch_normalization_3/Const*
dtype0
�
/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
dtype0*
_output_shapes	
:�
l
batch_normalization_3/Const_1Const*
valueB�*    *
dtype0*
_output_shapes	
:�
�
batch_normalization_3/betaVarHandleOp*
shape:�*
dtype0*
_output_shapes
: *+
shared_namebatch_normalization_3/beta*-
_class#
!loc:@batch_normalization_3/beta*
	container 
�
;batch_normalization_3/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_3/beta*
_output_shapes
: 
}
!batch_normalization_3/beta/AssignAssignVariableOpbatch_normalization_3/betabatch_normalization_3/Const_1*
dtype0
�
.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
dtype0*
_output_shapes	
:�
l
batch_normalization_3/Const_2Const*
valueB�*    *
dtype0*
_output_shapes	
:�
�
!batch_normalization_3/moving_meanVarHandleOp*
dtype0*
_output_shapes
: *2
shared_name#!batch_normalization_3/moving_mean*4
_class*
(&loc:@batch_normalization_3/moving_mean*
	container *
shape:�
�
Bbatch_normalization_3/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp!batch_normalization_3/moving_mean*
_output_shapes
: 
�
(batch_normalization_3/moving_mean/AssignAssignVariableOp!batch_normalization_3/moving_meanbatch_normalization_3/Const_2*
dtype0
�
5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
dtype0*
_output_shapes	
:�
l
batch_normalization_3/Const_3Const*
valueB�*  �?*
dtype0*
_output_shapes	
:�
�
%batch_normalization_3/moving_varianceVarHandleOp*
	container *
shape:�*
dtype0*
_output_shapes
: *6
shared_name'%batch_normalization_3/moving_variance*8
_class.
,*loc:@batch_normalization_3/moving_variance
�
Fbatch_normalization_3/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp%batch_normalization_3/moving_variance*
_output_shapes
: 
�
,batch_normalization_3/moving_variance/AssignAssignVariableOp%batch_normalization_3/moving_variancebatch_normalization_3/Const_3*
dtype0
�
9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
dtype0*
_output_shapes	
:�
~
4batch_normalization_3/moments/mean/reduction_indicesConst*
valueB: *
dtype0*
_output_shapes
:
�
"batch_normalization_3/moments/meanMeandense_3/Relu4batch_normalization_3/moments/mean/reduction_indices*
T0*
_output_shapes
:	�*
	keep_dims(*

Tidx0
�
*batch_normalization_3/moments/StopGradientStopGradient"batch_normalization_3/moments/mean*
T0*
_output_shapes
:	�
�
/batch_normalization_3/moments/SquaredDifferenceSquaredDifferencedense_3/Relu*batch_normalization_3/moments/StopGradient*(
_output_shapes
:����������*
T0
�
8batch_normalization_3/moments/variance/reduction_indicesConst*
valueB: *
dtype0*
_output_shapes
:
�
&batch_normalization_3/moments/varianceMean/batch_normalization_3/moments/SquaredDifference8batch_normalization_3/moments/variance/reduction_indices*
T0*
_output_shapes
:	�*
	keep_dims(*

Tidx0
�
%batch_normalization_3/moments/SqueezeSqueeze"batch_normalization_3/moments/mean*
T0*
_output_shapes	
:�*
squeeze_dims
 
�
'batch_normalization_3/moments/Squeeze_1Squeeze&batch_normalization_3/moments/variance*
squeeze_dims
 *
T0*
_output_shapes	
:�
j
%batch_normalization_3/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
#batch_normalization_3/batchnorm/addAddV2'batch_normalization_3/moments/Squeeze_1%batch_normalization_3/batchnorm/add/y*
T0*
_output_shapes	
:�
y
%batch_normalization_3/batchnorm/RsqrtRsqrt#batch_normalization_3/batchnorm/add*
T0*
_output_shapes	
:�
�
2batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
dtype0*
_output_shapes	
:�
�
#batch_normalization_3/batchnorm/mulMul%batch_normalization_3/batchnorm/Rsqrt2batch_normalization_3/batchnorm/mul/ReadVariableOp*
T0*
_output_shapes	
:�
�
%batch_normalization_3/batchnorm/mul_1Muldense_3/Relu#batch_normalization_3/batchnorm/mul*
T0*(
_output_shapes
:����������
�
%batch_normalization_3/batchnorm/mul_2Mul%batch_normalization_3/moments/Squeeze#batch_normalization_3/batchnorm/mul*
_output_shapes	
:�*
T0
�
.batch_normalization_3/batchnorm/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
dtype0*
_output_shapes	
:�
�
#batch_normalization_3/batchnorm/subSub.batch_normalization_3/batchnorm/ReadVariableOp%batch_normalization_3/batchnorm/mul_2*
T0*
_output_shapes	
:�
�
%batch_normalization_3/batchnorm/add_1AddV2%batch_normalization_3/batchnorm/mul_1#batch_normalization_3/batchnorm/sub*
T0*(
_output_shapes
:����������
g
batch_normalization_3/ShapeShapedense_3/Relu*
T0*
out_type0*
_output_shapes
:
s
)batch_normalization_3/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
u
+batch_normalization_3/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
u
+batch_normalization_3/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
#batch_normalization_3/strided_sliceStridedSlicebatch_normalization_3/Shape)batch_normalization_3/strided_slice/stack+batch_normalization_3/strided_slice/stack_1+batch_normalization_3/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
�
!batch_normalization_3/Rank/packedPack#batch_normalization_3/strided_slice*
T0*

axis *
N*
_output_shapes
:
\
batch_normalization_3/RankConst*
value	B :*
dtype0*
_output_shapes
: 
c
!batch_normalization_3/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
c
!batch_normalization_3/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
batch_normalization_3/rangeRange!batch_normalization_3/range/startbatch_normalization_3/Rank!batch_normalization_3/range/delta*
_output_shapes
:*

Tidx0
�
 batch_normalization_3/Prod/inputPack#batch_normalization_3/strided_slice*
T0*

axis *
N*
_output_shapes
:
�
batch_normalization_3/ProdProd batch_normalization_3/Prod/inputbatch_normalization_3/range*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
~
batch_normalization_3/CastCastbatch_normalization_3/Prod*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
`
batch_normalization_3/sub/yConst*
valueB
 *� �?*
dtype0*
_output_shapes
: 
z
batch_normalization_3/subSubbatch_normalization_3/Castbatch_normalization_3/sub/y*
T0*
_output_shapes
: 
�
batch_normalization_3/truedivRealDivbatch_normalization_3/Castbatch_normalization_3/sub*
T0*
_output_shapes
: 
�
batch_normalization_3/mulMul'batch_normalization_3/moments/Squeeze_1batch_normalization_3/truediv*
T0*
_output_shapes	
:�
�
batch_normalization_3/Const_4Const*
valueB
 *
�#<*4
_class*
(&loc:@batch_normalization_3/moving_mean*
dtype0*
_output_shapes
: 
�
$batch_normalization_3/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
dtype0*
_output_shapes	
:�
�
batch_normalization_3/sub_1Sub$batch_normalization_3/ReadVariableOp%batch_normalization_3/moments/Squeeze*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
_output_shapes	
:�
�
batch_normalization_3/mul_1Mulbatch_normalization_3/sub_1batch_normalization_3/Const_4*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
_output_shapes	
:�
�
)batch_normalization_3/AssignSubVariableOpAssignSubVariableOp!batch_normalization_3/moving_meanbatch_normalization_3/mul_1*4
_class*
(&loc:@batch_normalization_3/moving_mean*
dtype0
�
&batch_normalization_3/ReadVariableOp_1ReadVariableOp!batch_normalization_3/moving_mean*^batch_normalization_3/AssignSubVariableOp*
dtype0*
_output_shapes	
:�*4
_class*
(&loc:@batch_normalization_3/moving_mean
�
batch_normalization_3/Const_5Const*
valueB
 *
�#<*8
_class.
,*loc:@batch_normalization_3/moving_variance*
dtype0*
_output_shapes
: 
�
&batch_normalization_3/ReadVariableOp_2ReadVariableOp%batch_normalization_3/moving_variance*
dtype0*
_output_shapes	
:�
�
batch_normalization_3/sub_2Sub&batch_normalization_3/ReadVariableOp_2batch_normalization_3/mul*
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes	
:�
�
batch_normalization_3/mul_2Mulbatch_normalization_3/sub_2batch_normalization_3/Const_5*
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes	
:�
�
+batch_normalization_3/AssignSubVariableOp_1AssignSubVariableOp%batch_normalization_3/moving_variancebatch_normalization_3/mul_2*
dtype0*8
_class.
,*loc:@batch_normalization_3/moving_variance
�
&batch_normalization_3/ReadVariableOp_3ReadVariableOp%batch_normalization_3/moving_variance,^batch_normalization_3/AssignSubVariableOp_1*
dtype0*
_output_shapes	
:�*8
_class.
,*loc:@batch_normalization_3/moving_variance
z
!batch_normalization_3/cond/SwitchSwitchkeras_learning_phasekeras_learning_phase*
T0
*
_output_shapes
: : 
u
#batch_normalization_3/cond/switch_tIdentity#batch_normalization_3/cond/Switch:1*
_output_shapes
: *
T0

s
#batch_normalization_3/cond/switch_fIdentity!batch_normalization_3/cond/Switch*
T0
*
_output_shapes
: 
e
"batch_normalization_3/cond/pred_idIdentitykeras_learning_phase*
T0
*
_output_shapes
: 
�
#batch_normalization_3/cond/Switch_1Switch%batch_normalization_3/batchnorm/add_1"batch_normalization_3/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1*<
_output_shapes*
(:����������:����������
�
3batch_normalization_3/cond/batchnorm/ReadVariableOpReadVariableOp:batch_normalization_3/cond/batchnorm/ReadVariableOp/Switch*
dtype0*
_output_shapes	
:�
�
:batch_normalization_3/cond/batchnorm/ReadVariableOp/SwitchSwitch%batch_normalization_3/moving_variance"batch_normalization_3/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes
: : 
�
*batch_normalization_3/cond/batchnorm/add/yConst$^batch_normalization_3/cond/switch_f*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
(batch_normalization_3/cond/batchnorm/addAddV23batch_normalization_3/cond/batchnorm/ReadVariableOp*batch_normalization_3/cond/batchnorm/add/y*
T0*
_output_shapes	
:�
�
*batch_normalization_3/cond/batchnorm/RsqrtRsqrt(batch_normalization_3/cond/batchnorm/add*
T0*
_output_shapes	
:�
�
7batch_normalization_3/cond/batchnorm/mul/ReadVariableOpReadVariableOp>batch_normalization_3/cond/batchnorm/mul/ReadVariableOp/Switch*
dtype0*
_output_shapes	
:�
�
>batch_normalization_3/cond/batchnorm/mul/ReadVariableOp/SwitchSwitchbatch_normalization_3/gamma"batch_normalization_3/cond/pred_id*
_output_shapes
: : *
T0*.
_class$
" loc:@batch_normalization_3/gamma
�
(batch_normalization_3/cond/batchnorm/mulMul*batch_normalization_3/cond/batchnorm/Rsqrt7batch_normalization_3/cond/batchnorm/mul/ReadVariableOp*
T0*
_output_shapes	
:�
�
*batch_normalization_3/cond/batchnorm/mul_1Mul1batch_normalization_3/cond/batchnorm/mul_1/Switch(batch_normalization_3/cond/batchnorm/mul*
T0*(
_output_shapes
:����������
�
1batch_normalization_3/cond/batchnorm/mul_1/SwitchSwitchdense_3/Relu"batch_normalization_3/cond/pred_id*
T0*
_class
loc:@dense_3/Relu*<
_output_shapes*
(:����������:����������
�
5batch_normalization_3/cond/batchnorm/ReadVariableOp_1ReadVariableOp<batch_normalization_3/cond/batchnorm/ReadVariableOp_1/Switch*
dtype0*
_output_shapes	
:�
�
<batch_normalization_3/cond/batchnorm/ReadVariableOp_1/SwitchSwitch!batch_normalization_3/moving_mean"batch_normalization_3/cond/pred_id*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
_output_shapes
: : 
�
*batch_normalization_3/cond/batchnorm/mul_2Mul5batch_normalization_3/cond/batchnorm/ReadVariableOp_1(batch_normalization_3/cond/batchnorm/mul*
T0*
_output_shapes	
:�
�
5batch_normalization_3/cond/batchnorm/ReadVariableOp_2ReadVariableOp<batch_normalization_3/cond/batchnorm/ReadVariableOp_2/Switch*
dtype0*
_output_shapes	
:�
�
<batch_normalization_3/cond/batchnorm/ReadVariableOp_2/SwitchSwitchbatch_normalization_3/beta"batch_normalization_3/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_3/beta*
_output_shapes
: : 
�
(batch_normalization_3/cond/batchnorm/subSub5batch_normalization_3/cond/batchnorm/ReadVariableOp_2*batch_normalization_3/cond/batchnorm/mul_2*
T0*
_output_shapes	
:�
�
*batch_normalization_3/cond/batchnorm/add_1AddV2*batch_normalization_3/cond/batchnorm/mul_1(batch_normalization_3/cond/batchnorm/sub*
T0*(
_output_shapes
:����������
�
 batch_normalization_3/cond/MergeMerge*batch_normalization_3/cond/batchnorm/add_1%batch_normalization_3/cond/Switch_1:1*
T0*
N**
_output_shapes
:����������: 
n
dropout_2/cond/SwitchSwitchkeras_learning_phasekeras_learning_phase*
T0
*
_output_shapes
: : 
]
dropout_2/cond/switch_tIdentitydropout_2/cond/Switch:1*
T0
*
_output_shapes
: 
[
dropout_2/cond/switch_fIdentitydropout_2/cond/Switch*
T0
*
_output_shapes
: 
Y
dropout_2/cond/pred_idIdentitykeras_learning_phase*
T0
*
_output_shapes
: 
z
dropout_2/cond/dropout/rateConst^dropout_2/cond/switch_t*
valueB
 *   ?*
dtype0*
_output_shapes
: 
�
dropout_2/cond/dropout/ShapeShape%dropout_2/cond/dropout/Shape/Switch:1*
T0*
out_type0*
_output_shapes
:
�
#dropout_2/cond/dropout/Shape/SwitchSwitch batch_normalization_3/cond/Mergedropout_2/cond/pred_id*
T0*3
_class)
'%loc:@batch_normalization_3/cond/Merge*<
_output_shapes*
(:����������:����������
�
)dropout_2/cond/dropout/random_uniform/minConst^dropout_2/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
�
)dropout_2/cond/dropout/random_uniform/maxConst^dropout_2/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
3dropout_2/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_2/cond/dropout/Shape*
T0*
dtype0*(
_output_shapes
:����������*
seed2�ڡ*
seed���)
�
)dropout_2/cond/dropout/random_uniform/subSub)dropout_2/cond/dropout/random_uniform/max)dropout_2/cond/dropout/random_uniform/min*
_output_shapes
: *
T0
�
)dropout_2/cond/dropout/random_uniform/mulMul3dropout_2/cond/dropout/random_uniform/RandomUniform)dropout_2/cond/dropout/random_uniform/sub*
T0*(
_output_shapes
:����������
�
%dropout_2/cond/dropout/random_uniformAdd)dropout_2/cond/dropout/random_uniform/mul)dropout_2/cond/dropout/random_uniform/min*(
_output_shapes
:����������*
T0
{
dropout_2/cond/dropout/sub/xConst^dropout_2/cond/switch_t*
valueB
 *  �?*
dtype0*
_output_shapes
: 
}
dropout_2/cond/dropout/subSubdropout_2/cond/dropout/sub/xdropout_2/cond/dropout/rate*
T0*
_output_shapes
: 

 dropout_2/cond/dropout/truediv/xConst^dropout_2/cond/switch_t*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
dropout_2/cond/dropout/truedivRealDiv dropout_2/cond/dropout/truediv/xdropout_2/cond/dropout/sub*
_output_shapes
: *
T0
�
#dropout_2/cond/dropout/GreaterEqualGreaterEqual%dropout_2/cond/dropout/random_uniformdropout_2/cond/dropout/rate*
T0*(
_output_shapes
:����������
�
dropout_2/cond/dropout/mulMul%dropout_2/cond/dropout/Shape/Switch:1dropout_2/cond/dropout/truediv*(
_output_shapes
:����������*
T0
�
dropout_2/cond/dropout/CastCast#dropout_2/cond/dropout/GreaterEqual*

SrcT0
*
Truncate( *(
_output_shapes
:����������*

DstT0
�
dropout_2/cond/dropout/mul_1Muldropout_2/cond/dropout/muldropout_2/cond/dropout/Cast*
T0*(
_output_shapes
:����������
�
dropout_2/cond/Switch_1Switch batch_normalization_3/cond/Mergedropout_2/cond/pred_id*<
_output_shapes*
(:����������:����������*
T0*3
_class)
'%loc:@batch_normalization_3/cond/Merge
�
dropout_2/cond/MergeMergedropout_2/cond/Switch_1dropout_2/cond/dropout/mul_1*
T0*
N**
_output_shapes
:����������: 
m
dense_4/random_uniform/shapeConst*
valueB"�   �   *
dtype0*
_output_shapes
:
_
dense_4/random_uniform/minConst*
valueB
 *q��*
dtype0*
_output_shapes
: 
_
dense_4/random_uniform/maxConst*
valueB
 *q�>*
dtype0*
_output_shapes
: 
�
$dense_4/random_uniform/RandomUniformRandomUniformdense_4/random_uniform/shape*
dtype0* 
_output_shapes
:
��*
seed2�*
seed���)*
T0
z
dense_4/random_uniform/subSubdense_4/random_uniform/maxdense_4/random_uniform/min*
T0*
_output_shapes
: 
�
dense_4/random_uniform/mulMul$dense_4/random_uniform/RandomUniformdense_4/random_uniform/sub*
T0* 
_output_shapes
:
��
�
dense_4/random_uniformAdddense_4/random_uniform/muldense_4/random_uniform/min*
T0* 
_output_shapes
:
��
�
dense_4/kernelVarHandleOp*
	container *
shape:
��*
dtype0*
_output_shapes
: *
shared_namedense_4/kernel*!
_class
loc:@dense_4/kernel
m
/dense_4/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_4/kernel*
_output_shapes
: 
^
dense_4/kernel/AssignAssignVariableOpdense_4/kerneldense_4/random_uniform*
dtype0
s
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
dtype0* 
_output_shapes
:
��
\
dense_4/ConstConst*
dtype0*
_output_shapes	
:�*
valueB�*    
�
dense_4/biasVarHandleOp*
dtype0*
_output_shapes
: *
shared_namedense_4/bias*
_class
loc:@dense_4/bias*
	container *
shape:�
i
-dense_4/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_4/bias*
_output_shapes
: 
Q
dense_4/bias/AssignAssignVariableOpdense_4/biasdense_4/Const*
dtype0
j
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
dtype0*
_output_shapes	
:�
n
dense_4/MatMul/ReadVariableOpReadVariableOpdense_4/kernel*
dtype0* 
_output_shapes
:
��
�
dense_4/MatMulMatMuldropout_2/cond/Mergedense_4/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b( 
h
dense_4/BiasAdd/ReadVariableOpReadVariableOpdense_4/bias*
dtype0*
_output_shapes	
:�
�
dense_4/BiasAddBiasAdddense_4/MatMuldense_4/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*(
_output_shapes
:����������
X
dense_4/ReluReludense_4/BiasAdd*
T0*(
_output_shapes
:����������
j
batch_normalization_4/ConstConst*
valueB�*  �?*
dtype0*
_output_shapes	
:�
�
batch_normalization_4/gammaVarHandleOp*,
shared_namebatch_normalization_4/gamma*.
_class$
" loc:@batch_normalization_4/gamma*
	container *
shape:�*
dtype0*
_output_shapes
: 
�
<batch_normalization_4/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_4/gamma*
_output_shapes
: 
}
"batch_normalization_4/gamma/AssignAssignVariableOpbatch_normalization_4/gammabatch_normalization_4/Const*
dtype0
�
/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_4/gamma*
dtype0*
_output_shapes	
:�
l
batch_normalization_4/Const_1Const*
dtype0*
_output_shapes	
:�*
valueB�*    
�
batch_normalization_4/betaVarHandleOp*-
_class#
!loc:@batch_normalization_4/beta*
	container *
shape:�*
dtype0*
_output_shapes
: *+
shared_namebatch_normalization_4/beta
�
;batch_normalization_4/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_4/beta*
_output_shapes
: 
}
!batch_normalization_4/beta/AssignAssignVariableOpbatch_normalization_4/betabatch_normalization_4/Const_1*
dtype0
�
.batch_normalization_4/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_4/beta*
dtype0*
_output_shapes	
:�
l
batch_normalization_4/Const_2Const*
valueB�*    *
dtype0*
_output_shapes	
:�
�
!batch_normalization_4/moving_meanVarHandleOp*
dtype0*
_output_shapes
: *2
shared_name#!batch_normalization_4/moving_mean*4
_class*
(&loc:@batch_normalization_4/moving_mean*
	container *
shape:�
�
Bbatch_normalization_4/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp!batch_normalization_4/moving_mean*
_output_shapes
: 
�
(batch_normalization_4/moving_mean/AssignAssignVariableOp!batch_normalization_4/moving_meanbatch_normalization_4/Const_2*
dtype0
�
5batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
dtype0*
_output_shapes	
:�
l
batch_normalization_4/Const_3Const*
valueB�*  �?*
dtype0*
_output_shapes	
:�
�
%batch_normalization_4/moving_varianceVarHandleOp*
shape:�*
dtype0*
_output_shapes
: *6
shared_name'%batch_normalization_4/moving_variance*8
_class.
,*loc:@batch_normalization_4/moving_variance*
	container 
�
Fbatch_normalization_4/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp%batch_normalization_4/moving_variance*
_output_shapes
: 
�
,batch_normalization_4/moving_variance/AssignAssignVariableOp%batch_normalization_4/moving_variancebatch_normalization_4/Const_3*
dtype0
�
9batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*
dtype0*
_output_shapes	
:�
~
4batch_normalization_4/moments/mean/reduction_indicesConst*
valueB: *
dtype0*
_output_shapes
:
�
"batch_normalization_4/moments/meanMeandense_4/Relu4batch_normalization_4/moments/mean/reduction_indices*
T0*
_output_shapes
:	�*
	keep_dims(*

Tidx0
�
*batch_normalization_4/moments/StopGradientStopGradient"batch_normalization_4/moments/mean*
T0*
_output_shapes
:	�
�
/batch_normalization_4/moments/SquaredDifferenceSquaredDifferencedense_4/Relu*batch_normalization_4/moments/StopGradient*(
_output_shapes
:����������*
T0
�
8batch_normalization_4/moments/variance/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB: 
�
&batch_normalization_4/moments/varianceMean/batch_normalization_4/moments/SquaredDifference8batch_normalization_4/moments/variance/reduction_indices*
T0*
_output_shapes
:	�*
	keep_dims(*

Tidx0
�
%batch_normalization_4/moments/SqueezeSqueeze"batch_normalization_4/moments/mean*
squeeze_dims
 *
T0*
_output_shapes	
:�
�
'batch_normalization_4/moments/Squeeze_1Squeeze&batch_normalization_4/moments/variance*
T0*
_output_shapes	
:�*
squeeze_dims
 
j
%batch_normalization_4/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
#batch_normalization_4/batchnorm/addAddV2'batch_normalization_4/moments/Squeeze_1%batch_normalization_4/batchnorm/add/y*
_output_shapes	
:�*
T0
y
%batch_normalization_4/batchnorm/RsqrtRsqrt#batch_normalization_4/batchnorm/add*
T0*
_output_shapes	
:�
�
2batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOpbatch_normalization_4/gamma*
dtype0*
_output_shapes	
:�
�
#batch_normalization_4/batchnorm/mulMul%batch_normalization_4/batchnorm/Rsqrt2batch_normalization_4/batchnorm/mul/ReadVariableOp*
T0*
_output_shapes	
:�
�
%batch_normalization_4/batchnorm/mul_1Muldense_4/Relu#batch_normalization_4/batchnorm/mul*
T0*(
_output_shapes
:����������
�
%batch_normalization_4/batchnorm/mul_2Mul%batch_normalization_4/moments/Squeeze#batch_normalization_4/batchnorm/mul*
_output_shapes	
:�*
T0
�
.batch_normalization_4/batchnorm/ReadVariableOpReadVariableOpbatch_normalization_4/beta*
dtype0*
_output_shapes	
:�
�
#batch_normalization_4/batchnorm/subSub.batch_normalization_4/batchnorm/ReadVariableOp%batch_normalization_4/batchnorm/mul_2*
_output_shapes	
:�*
T0
�
%batch_normalization_4/batchnorm/add_1AddV2%batch_normalization_4/batchnorm/mul_1#batch_normalization_4/batchnorm/sub*
T0*(
_output_shapes
:����������
g
batch_normalization_4/ShapeShapedense_4/Relu*
T0*
out_type0*
_output_shapes
:
s
)batch_normalization_4/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
u
+batch_normalization_4/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
u
+batch_normalization_4/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
#batch_normalization_4/strided_sliceStridedSlicebatch_normalization_4/Shape)batch_normalization_4/strided_slice/stack+batch_normalization_4/strided_slice/stack_1+batch_normalization_4/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
�
!batch_normalization_4/Rank/packedPack#batch_normalization_4/strided_slice*
T0*

axis *
N*
_output_shapes
:
\
batch_normalization_4/RankConst*
value	B :*
dtype0*
_output_shapes
: 
c
!batch_normalization_4/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
c
!batch_normalization_4/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
batch_normalization_4/rangeRange!batch_normalization_4/range/startbatch_normalization_4/Rank!batch_normalization_4/range/delta*

Tidx0*
_output_shapes
:
�
 batch_normalization_4/Prod/inputPack#batch_normalization_4/strided_slice*
T0*

axis *
N*
_output_shapes
:
�
batch_normalization_4/ProdProd batch_normalization_4/Prod/inputbatch_normalization_4/range*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
~
batch_normalization_4/CastCastbatch_normalization_4/Prod*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
`
batch_normalization_4/sub/yConst*
valueB
 *� �?*
dtype0*
_output_shapes
: 
z
batch_normalization_4/subSubbatch_normalization_4/Castbatch_normalization_4/sub/y*
T0*
_output_shapes
: 
�
batch_normalization_4/truedivRealDivbatch_normalization_4/Castbatch_normalization_4/sub*
T0*
_output_shapes
: 
�
batch_normalization_4/mulMul'batch_normalization_4/moments/Squeeze_1batch_normalization_4/truediv*
T0*
_output_shapes	
:�
�
batch_normalization_4/Const_4Const*
dtype0*
_output_shapes
: *
valueB
 *
�#<*4
_class*
(&loc:@batch_normalization_4/moving_mean
�
$batch_normalization_4/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
dtype0*
_output_shapes	
:�
�
batch_normalization_4/sub_1Sub$batch_normalization_4/ReadVariableOp%batch_normalization_4/moments/Squeeze*
_output_shapes	
:�*
T0*4
_class*
(&loc:@batch_normalization_4/moving_mean
�
batch_normalization_4/mul_1Mulbatch_normalization_4/sub_1batch_normalization_4/Const_4*
T0*4
_class*
(&loc:@batch_normalization_4/moving_mean*
_output_shapes	
:�
�
)batch_normalization_4/AssignSubVariableOpAssignSubVariableOp!batch_normalization_4/moving_meanbatch_normalization_4/mul_1*4
_class*
(&loc:@batch_normalization_4/moving_mean*
dtype0
�
&batch_normalization_4/ReadVariableOp_1ReadVariableOp!batch_normalization_4/moving_mean*^batch_normalization_4/AssignSubVariableOp*4
_class*
(&loc:@batch_normalization_4/moving_mean*
dtype0*
_output_shapes	
:�
�
batch_normalization_4/Const_5Const*
dtype0*
_output_shapes
: *
valueB
 *
�#<*8
_class.
,*loc:@batch_normalization_4/moving_variance
�
&batch_normalization_4/ReadVariableOp_2ReadVariableOp%batch_normalization_4/moving_variance*
dtype0*
_output_shapes	
:�
�
batch_normalization_4/sub_2Sub&batch_normalization_4/ReadVariableOp_2batch_normalization_4/mul*
_output_shapes	
:�*
T0*8
_class.
,*loc:@batch_normalization_4/moving_variance
�
batch_normalization_4/mul_2Mulbatch_normalization_4/sub_2batch_normalization_4/Const_5*
T0*8
_class.
,*loc:@batch_normalization_4/moving_variance*
_output_shapes	
:�
�
+batch_normalization_4/AssignSubVariableOp_1AssignSubVariableOp%batch_normalization_4/moving_variancebatch_normalization_4/mul_2*8
_class.
,*loc:@batch_normalization_4/moving_variance*
dtype0
�
&batch_normalization_4/ReadVariableOp_3ReadVariableOp%batch_normalization_4/moving_variance,^batch_normalization_4/AssignSubVariableOp_1*8
_class.
,*loc:@batch_normalization_4/moving_variance*
dtype0*
_output_shapes	
:�
z
!batch_normalization_4/cond/SwitchSwitchkeras_learning_phasekeras_learning_phase*
T0
*
_output_shapes
: : 
u
#batch_normalization_4/cond/switch_tIdentity#batch_normalization_4/cond/Switch:1*
T0
*
_output_shapes
: 
s
#batch_normalization_4/cond/switch_fIdentity!batch_normalization_4/cond/Switch*
_output_shapes
: *
T0

e
"batch_normalization_4/cond/pred_idIdentitykeras_learning_phase*
T0
*
_output_shapes
: 
�
#batch_normalization_4/cond/Switch_1Switch%batch_normalization_4/batchnorm/add_1"batch_normalization_4/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1*<
_output_shapes*
(:����������:����������
�
3batch_normalization_4/cond/batchnorm/ReadVariableOpReadVariableOp:batch_normalization_4/cond/batchnorm/ReadVariableOp/Switch*
dtype0*
_output_shapes	
:�
�
:batch_normalization_4/cond/batchnorm/ReadVariableOp/SwitchSwitch%batch_normalization_4/moving_variance"batch_normalization_4/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_4/moving_variance*
_output_shapes
: : 
�
*batch_normalization_4/cond/batchnorm/add/yConst$^batch_normalization_4/cond/switch_f*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
(batch_normalization_4/cond/batchnorm/addAddV23batch_normalization_4/cond/batchnorm/ReadVariableOp*batch_normalization_4/cond/batchnorm/add/y*
T0*
_output_shapes	
:�
�
*batch_normalization_4/cond/batchnorm/RsqrtRsqrt(batch_normalization_4/cond/batchnorm/add*
T0*
_output_shapes	
:�
�
7batch_normalization_4/cond/batchnorm/mul/ReadVariableOpReadVariableOp>batch_normalization_4/cond/batchnorm/mul/ReadVariableOp/Switch*
dtype0*
_output_shapes	
:�
�
>batch_normalization_4/cond/batchnorm/mul/ReadVariableOp/SwitchSwitchbatch_normalization_4/gamma"batch_normalization_4/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_4/gamma*
_output_shapes
: : 
�
(batch_normalization_4/cond/batchnorm/mulMul*batch_normalization_4/cond/batchnorm/Rsqrt7batch_normalization_4/cond/batchnorm/mul/ReadVariableOp*
_output_shapes	
:�*
T0
�
*batch_normalization_4/cond/batchnorm/mul_1Mul1batch_normalization_4/cond/batchnorm/mul_1/Switch(batch_normalization_4/cond/batchnorm/mul*
T0*(
_output_shapes
:����������
�
1batch_normalization_4/cond/batchnorm/mul_1/SwitchSwitchdense_4/Relu"batch_normalization_4/cond/pred_id*<
_output_shapes*
(:����������:����������*
T0*
_class
loc:@dense_4/Relu
�
5batch_normalization_4/cond/batchnorm/ReadVariableOp_1ReadVariableOp<batch_normalization_4/cond/batchnorm/ReadVariableOp_1/Switch*
dtype0*
_output_shapes	
:�
�
<batch_normalization_4/cond/batchnorm/ReadVariableOp_1/SwitchSwitch!batch_normalization_4/moving_mean"batch_normalization_4/cond/pred_id*
T0*4
_class*
(&loc:@batch_normalization_4/moving_mean*
_output_shapes
: : 
�
*batch_normalization_4/cond/batchnorm/mul_2Mul5batch_normalization_4/cond/batchnorm/ReadVariableOp_1(batch_normalization_4/cond/batchnorm/mul*
T0*
_output_shapes	
:�
�
5batch_normalization_4/cond/batchnorm/ReadVariableOp_2ReadVariableOp<batch_normalization_4/cond/batchnorm/ReadVariableOp_2/Switch*
dtype0*
_output_shapes	
:�
�
<batch_normalization_4/cond/batchnorm/ReadVariableOp_2/SwitchSwitchbatch_normalization_4/beta"batch_normalization_4/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_4/beta*
_output_shapes
: : 
�
(batch_normalization_4/cond/batchnorm/subSub5batch_normalization_4/cond/batchnorm/ReadVariableOp_2*batch_normalization_4/cond/batchnorm/mul_2*
T0*
_output_shapes	
:�
�
*batch_normalization_4/cond/batchnorm/add_1AddV2*batch_normalization_4/cond/batchnorm/mul_1(batch_normalization_4/cond/batchnorm/sub*
T0*(
_output_shapes
:����������
�
 batch_normalization_4/cond/MergeMerge*batch_normalization_4/cond/batchnorm/add_1%batch_normalization_4/cond/Switch_1:1*
N**
_output_shapes
:����������: *
T0
m
dense_5/random_uniform/shapeConst*
valueB"�   +   *
dtype0*
_output_shapes
:
_
dense_5/random_uniform/minConst*
valueB
 *�?�*
dtype0*
_output_shapes
: 
_
dense_5/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *�?>
�
$dense_5/random_uniform/RandomUniformRandomUniformdense_5/random_uniform/shape*
T0*
dtype0*
_output_shapes
:	�+*
seed2�ب*
seed���)
z
dense_5/random_uniform/subSubdense_5/random_uniform/maxdense_5/random_uniform/min*
T0*
_output_shapes
: 
�
dense_5/random_uniform/mulMul$dense_5/random_uniform/RandomUniformdense_5/random_uniform/sub*
T0*
_output_shapes
:	�+

dense_5/random_uniformAdddense_5/random_uniform/muldense_5/random_uniform/min*
_output_shapes
:	�+*
T0
�
dense_5/kernelVarHandleOp*
	container *
shape:	�+*
dtype0*
_output_shapes
: *
shared_namedense_5/kernel*!
_class
loc:@dense_5/kernel
m
/dense_5/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_5/kernel*
_output_shapes
: 
^
dense_5/kernel/AssignAssignVariableOpdense_5/kerneldense_5/random_uniform*
dtype0
r
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
dtype0*
_output_shapes
:	�+
Z
dense_5/ConstConst*
valueB+*    *
dtype0*
_output_shapes
:+
�
dense_5/biasVarHandleOp*
dtype0*
_output_shapes
: *
shared_namedense_5/bias*
_class
loc:@dense_5/bias*
	container *
shape:+
i
-dense_5/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_5/bias*
_output_shapes
: 
Q
dense_5/bias/AssignAssignVariableOpdense_5/biasdense_5/Const*
dtype0
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
dtype0*
_output_shapes
:+
m
dense_5/MatMul/ReadVariableOpReadVariableOpdense_5/kernel*
dtype0*
_output_shapes
:	�+
�
dense_5/MatMulMatMul batch_normalization_4/cond/Mergedense_5/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������+*
transpose_a( *
transpose_b( 
g
dense_5/BiasAdd/ReadVariableOpReadVariableOpdense_5/bias*
dtype0*
_output_shapes
:+
�
dense_5/BiasAddBiasAdddense_5/MatMuldense_5/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*'
_output_shapes
:���������+
]
dense_5/SoftmaxSoftmaxdense_5/BiasAdd*
T0*'
_output_shapes
:���������+
�
)Adam/iterations/Initializer/initial_valueConst*
dtype0	*
_output_shapes
: *
value	B	 R *"
_class
loc:@Adam/iterations
�
Adam/iterationsVarHandleOp* 
shared_nameAdam/iterations*"
_class
loc:@Adam/iterations*
	container *
shape: *
dtype0	*
_output_shapes
: 
o
0Adam/iterations/IsInitialized/VarIsInitializedOpVarIsInitializedOpAdam/iterations*
_output_shapes
: 
s
Adam/iterations/AssignAssignVariableOpAdam/iterations)Adam/iterations/Initializer/initial_value*
dtype0	
k
#Adam/iterations/Read/ReadVariableOpReadVariableOpAdam/iterations*
dtype0	*
_output_shapes
: 
�
,Adam/learning_rate/Initializer/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *o�:*%
_class
loc:@Adam/learning_rate
�
Adam/learning_rateVarHandleOp*
dtype0*
_output_shapes
: *#
shared_nameAdam/learning_rate*%
_class
loc:@Adam/learning_rate*
	container *
shape: 
u
3Adam/learning_rate/IsInitialized/VarIsInitializedOpVarIsInitializedOpAdam/learning_rate*
_output_shapes
: 
|
Adam/learning_rate/AssignAssignVariableOpAdam/learning_rate,Adam/learning_rate/Initializer/initial_value*
dtype0
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
dtype0*
_output_shapes
: 
�
%Adam/beta_1/Initializer/initial_valueConst*
valueB
 *fff?*
_class
loc:@Adam/beta_1*
dtype0*
_output_shapes
: 
�
Adam/beta_1VarHandleOp*
dtype0*
_output_shapes
: *
shared_nameAdam/beta_1*
_class
loc:@Adam/beta_1*
	container *
shape: 
g
,Adam/beta_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpAdam/beta_1*
_output_shapes
: 
g
Adam/beta_1/AssignAssignVariableOpAdam/beta_1%Adam/beta_1/Initializer/initial_value*
dtype0
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
�
%Adam/beta_2/Initializer/initial_valueConst*
valueB
 *w�?*
_class
loc:@Adam/beta_2*
dtype0*
_output_shapes
: 
�
Adam/beta_2VarHandleOp*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_nameAdam/beta_2*
_class
loc:@Adam/beta_2
g
,Adam/beta_2/IsInitialized/VarIsInitializedOpVarIsInitializedOpAdam/beta_2*
_output_shapes
: 
g
Adam/beta_2/AssignAssignVariableOpAdam/beta_2%Adam/beta_2/Initializer/initial_value*
dtype0
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
�
$Adam/decay/Initializer/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *    *
_class
loc:@Adam/decay
�

Adam/decayVarHandleOp*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name
Adam/decay*
_class
loc:@Adam/decay
e
+Adam/decay/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Adam/decay*
_output_shapes
: 
d
Adam/decay/AssignAssignVariableOp
Adam/decay$Adam/decay/Initializer/initial_value*
dtype0
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
dtype0*
_output_shapes
: 
�
dense_5_targetPlaceholder*
dtype0*0
_output_shapes
:������������������*%
shape:������������������
q
dense_5_sample_weightsPlaceholder*
dtype0*#
_output_shapes
:���������*
shape:���������
J
ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
totalVarHandleOp*
dtype0*
_output_shapes
: *
shared_nametotal*
_class

loc:@total*
	container *
shape: 
[
&total/IsInitialized/VarIsInitializedOpVarIsInitializedOptotal*
_output_shapes
: 
;
total/AssignAssignVariableOptotalConst*
dtype0
W
total/Read/ReadVariableOpReadVariableOptotal*
dtype0*
_output_shapes
: 
L
Const_1Const*
valueB
 *    *
dtype0*
_output_shapes
: 
�
countVarHandleOp*
_class

loc:@count*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_namecount
[
&count/IsInitialized/VarIsInitializedOpVarIsInitializedOpcount*
_output_shapes
: 
=
count/AssignAssignVariableOpcountConst_1*
dtype0
W
count/Read/ReadVariableOpReadVariableOpcount*
dtype0*
_output_shapes
: 
l
!metrics/accuracy/ArgMax/dimensionConst*
dtype0*
_output_shapes
: *
valueB :
���������
�
metrics/accuracy/ArgMaxArgMaxdense_5_target!metrics/accuracy/ArgMax/dimension*
T0*
output_type0	*#
_output_shapes
:���������*

Tidx0
n
#metrics/accuracy/ArgMax_1/dimensionConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
metrics/accuracy/ArgMax_1ArgMaxdense_5/Softmax#metrics/accuracy/ArgMax_1/dimension*

Tidx0*
T0*
output_type0	*#
_output_shapes
:���������
�
metrics/accuracy/EqualEqualmetrics/accuracy/ArgMaxmetrics/accuracy/ArgMax_1*
T0	*#
_output_shapes
:���������*
incompatible_shape_error(
�
metrics/accuracy/CastCastmetrics/accuracy/Equal*

SrcT0
*
Truncate( *#
_output_shapes
:���������*

DstT0
`
metrics/accuracy/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
�
metrics/accuracy/SumSummetrics/accuracy/Castmetrics/accuracy/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
e
$metrics/accuracy/AssignAddVariableOpAssignAddVariableOptotalmetrics/accuracy/Sum*
dtype0
�
metrics/accuracy/ReadVariableOpReadVariableOptotal%^metrics/accuracy/AssignAddVariableOp*
dtype0*
_output_shapes
: 
e
metrics/accuracy/SizeSizemetrics/accuracy/Cast*
T0*
out_type0*
_output_shapes
: 
v
metrics/accuracy/Cast_1Castmetrics/accuracy/Size*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
j
&metrics/accuracy/AssignAddVariableOp_1AssignAddVariableOpcountmetrics/accuracy/Cast_1*
dtype0
�
!metrics/accuracy/ReadVariableOp_1ReadVariableOpcount'^metrics/accuracy/AssignAddVariableOp_1*
dtype0*
_output_shapes
: 
�
!metrics/accuracy/ReadVariableOp_2ReadVariableOptotal%^metrics/accuracy/AssignAddVariableOp'^metrics/accuracy/AssignAddVariableOp_1*
dtype0*
_output_shapes
: 
�
'metrics/accuracy/truediv/ReadVariableOpReadVariableOpcount%^metrics/accuracy/AssignAddVariableOp'^metrics/accuracy/AssignAddVariableOp_1*
dtype0*
_output_shapes
: 
�
metrics/accuracy/truedivRealDiv!metrics/accuracy/ReadVariableOp_2'metrics/accuracy/truediv/ReadVariableOp*
T0*
_output_shapes
: 
`
metrics/accuracy/IdentityIdentitymetrics/accuracy/truediv*
T0*
_output_shapes
: 
�
Qloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/RankConst*
value	B :*
dtype0*
_output_shapes
: 
�
Rloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/ShapeShapedense_5/BiasAdd*
_output_shapes
:*
T0*
out_type0
�
Sloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
�
Tloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Shape_1Shapedense_5/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
Rloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
Ploss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/SubSubSloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Rank_1Rloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Sub/y*
T0*
_output_shapes
: 
�
Xloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Slice/beginPackPloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Sub*
T0*

axis *
N*
_output_shapes
:
�
Wloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
�
Rloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/SliceSliceTloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Shape_1Xloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Slice/beginWloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Slice/size*
Index0*
T0*
_output_shapes
:
�
\loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/concat/values_0Const*
dtype0*
_output_shapes
:*
valueB:
���������
�
Xloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Sloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/concatConcatV2\loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/concat/values_0Rloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/SliceXloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
�
Tloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/ReshapeReshapedense_5/BiasAddSloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/concat*
T0*
Tshape0*0
_output_shapes
:������������������
�
Sloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Rank_2Const*
value	B :*
dtype0*
_output_shapes
: 
�
Tloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Shape_2Shapedense_5_target*
T0*
out_type0*
_output_shapes
:
�
Tloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Sub_1/yConst*
dtype0*
_output_shapes
: *
value	B :
�
Rloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Sub_1SubSloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Rank_2Tloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Sub_1/y*
T0*
_output_shapes
: 
�
Zloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Slice_1/beginPackRloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Sub_1*
T0*

axis *
N*
_output_shapes
:
�
Yloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
�
Tloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Slice_1SliceTloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Shape_2Zloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Slice_1/beginYloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Slice_1/size*
Index0*
T0*
_output_shapes
:
�
^loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/concat_1/values_0Const*
valueB:
���������*
dtype0*
_output_shapes
:
�
Zloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Uloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/concat_1ConcatV2^loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/concat_1/values_0Tloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Slice_1Zloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:
�
Vloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape_1Reshapedense_5_targetUloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/concat_1*
T0*
Tshape0*0
_output_shapes
:������������������
�
Lloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogitsTloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/ReshapeVloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape_1*
T0*?
_output_shapes-
+:���������:������������������
�
Tloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
Rloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Sub_2SubQloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/RankTloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Sub_2/y*
T0*
_output_shapes
: 
�
Zloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Slice_2/beginConst*
valueB: *
dtype0*
_output_shapes
:
�
Yloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Slice_2/sizePackRloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Sub_2*
T0*

axis *
N*
_output_shapes
:
�
Tloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Slice_2SliceRloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/ShapeZloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Slice_2/beginYloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Slice_2/size*
Index0*
T0*
_output_shapes
:
�
Vloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape_2ReshapeLloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logitsTloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Slice_2*#
_output_shapes
:���������*
T0*
Tshape0
�
<loss/dense_5_loss/categorical_crossentropy/weighted_loss/mulMuldense_5_sample_weightsVloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape_2*
T0*#
_output_shapes
:���������
�
>loss/dense_5_loss/categorical_crossentropy/weighted_loss/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
�
<loss/dense_5_loss/categorical_crossentropy/weighted_loss/SumSum<loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul>loss/dense_5_loss/categorical_crossentropy/weighted_loss/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
Jloss/dense_5_loss/categorical_crossentropy/weighted_loss/num_elements/SizeSize<loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul*
_output_shapes
: *
T0*
out_type0
�
Jloss/dense_5_loss/categorical_crossentropy/weighted_loss/num_elements/CastCastJloss/dense_5_loss/categorical_crossentropy/weighted_loss/num_elements/Size*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0
�
@loss/dense_5_loss/categorical_crossentropy/weighted_loss/truedivRealDiv<loss/dense_5_loss/categorical_crossentropy/weighted_loss/SumJloss/dense_5_loss/categorical_crossentropy/weighted_loss/num_elements/Cast*
T0*
_output_shapes
: 
O

loss/mul/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
~
loss/mulMul
loss/mul/x@loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv*
T0*
_output_shapes
: 
J
Const_2Const*
valueB *
dtype0*
_output_shapes
: 
]
MeanMeanloss/mulConst_2*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
y
training/Adam/gradients/ShapeConst*
dtype0*
_output_shapes
: *
valueB *
_class
	loc:@Mean

!training/Adam/gradients/grad_ys_0Const*
valueB
 *  �?*
_class
	loc:@Mean*
dtype0*
_output_shapes
: 
�
training/Adam/gradients/FillFilltraining/Adam/gradients/Shape!training/Adam/gradients/grad_ys_0*
T0*

index_type0*
_class
	loc:@Mean*
_output_shapes
: 
�
/training/Adam/gradients/Mean_grad/Reshape/shapeConst*
valueB *
_class
	loc:@Mean*
dtype0*
_output_shapes
: 
�
)training/Adam/gradients/Mean_grad/ReshapeReshapetraining/Adam/gradients/Fill/training/Adam/gradients/Mean_grad/Reshape/shape*
_output_shapes
: *
T0*
Tshape0*
_class
	loc:@Mean
�
'training/Adam/gradients/Mean_grad/ConstConst*
dtype0*
_output_shapes
: *
valueB *
_class
	loc:@Mean
�
&training/Adam/gradients/Mean_grad/TileTile)training/Adam/gradients/Mean_grad/Reshape'training/Adam/gradients/Mean_grad/Const*

Tmultiples0*
T0*
_class
	loc:@Mean*
_output_shapes
: 
�
)training/Adam/gradients/Mean_grad/Const_1Const*
dtype0*
_output_shapes
: *
valueB
 *  �?*
_class
	loc:@Mean
�
)training/Adam/gradients/Mean_grad/truedivRealDiv&training/Adam/gradients/Mean_grad/Tile)training/Adam/gradients/Mean_grad/Const_1*
T0*
_class
	loc:@Mean*
_output_shapes
: 
�
)training/Adam/gradients/loss/mul_grad/MulMul)training/Adam/gradients/Mean_grad/truediv@loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv*
T0*
_class
loc:@loss/mul*
_output_shapes
: 
�
+training/Adam/gradients/loss/mul_grad/Mul_1Mul)training/Adam/gradients/Mean_grad/truediv
loss/mul/x*
_output_shapes
: *
T0*
_class
loc:@loss/mul
�
ctraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/ShapeConst*
valueB *S
_classI
GEloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv*
dtype0*
_output_shapes
: 
�
etraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/Shape_1Const*
valueB *S
_classI
GEloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv*
dtype0*
_output_shapes
: 
�
straining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsctraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/Shapeetraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/Shape_1*
T0*S
_classI
GEloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv*2
_output_shapes 
:���������:���������
�
etraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/RealDivRealDiv+training/Adam/gradients/loss/mul_grad/Mul_1Jloss/dense_5_loss/categorical_crossentropy/weighted_loss/num_elements/Cast*
_output_shapes
: *
T0*S
_classI
GEloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv
�
atraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/SumSumetraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/RealDivstraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/BroadcastGradientArgs*
T0*S
_classI
GEloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
etraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/ReshapeReshapeatraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/Sumctraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/Shape*
T0*
Tshape0*S
_classI
GEloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv*
_output_shapes
: 
�
atraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/NegNeg<loss/dense_5_loss/categorical_crossentropy/weighted_loss/Sum*
T0*S
_classI
GEloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv*
_output_shapes
: 
�
gtraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/RealDiv_1RealDivatraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/NegJloss/dense_5_loss/categorical_crossentropy/weighted_loss/num_elements/Cast*
T0*S
_classI
GEloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv*
_output_shapes
: 
�
gtraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/RealDiv_2RealDivgtraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/RealDiv_1Jloss/dense_5_loss/categorical_crossentropy/weighted_loss/num_elements/Cast*
T0*S
_classI
GEloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv*
_output_shapes
: 
�
atraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/mulMul+training/Adam/gradients/loss/mul_grad/Mul_1gtraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/RealDiv_2*
T0*S
_classI
GEloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv*
_output_shapes
: 
�
ctraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/Sum_1Sumatraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/mulutraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/BroadcastGradientArgs:1*
T0*S
_classI
GEloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
gtraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/Reshape_1Reshapectraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/Sum_1etraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/Shape_1*
T0*
Tshape0*S
_classI
GEloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv*
_output_shapes
: 
�
gtraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/Sum_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:*O
_classE
CAloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/Sum
�
atraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/Sum_grad/ReshapeReshapeetraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/Reshapegtraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/Sum_grad/Reshape/shape*
T0*
Tshape0*O
_classE
CAloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/Sum*
_output_shapes
:
�
_training/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/Sum_grad/ShapeShape<loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul*
T0*
out_type0*O
_classE
CAloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/Sum*
_output_shapes
:
�
^training/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/Sum_grad/TileTileatraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/Sum_grad/Reshape_training/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/Sum_grad/Shape*

Tmultiples0*
T0*O
_classE
CAloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/Sum*#
_output_shapes
:���������
�
_training/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul_grad/ShapeShapedense_5_sample_weights*
T0*
out_type0*O
_classE
CAloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul*
_output_shapes
:
�
atraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul_grad/Shape_1ShapeVloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape_2*
T0*
out_type0*O
_classE
CAloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul*
_output_shapes
:
�
otraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs_training/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul_grad/Shapeatraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul_grad/Shape_1*
T0*O
_classE
CAloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul*2
_output_shapes 
:���������:���������
�
]training/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul_grad/MulMul^training/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/Sum_grad/TileVloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape_2*
T0*O
_classE
CAloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul*#
_output_shapes
:���������
�
]training/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul_grad/SumSum]training/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul_grad/Mulotraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*O
_classE
CAloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul
�
atraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul_grad/ReshapeReshape]training/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul_grad/Sum_training/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul_grad/Shape*#
_output_shapes
:���������*
T0*
Tshape0*O
_classE
CAloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul
�
_training/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul_grad/Mul_1Muldense_5_sample_weights^training/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/Sum_grad/Tile*
T0*O
_classE
CAloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul*#
_output_shapes
:���������
�
_training/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul_grad/Sum_1Sum_training/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul_grad/Mul_1qtraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul_grad/BroadcastGradientArgs:1*
T0*O
_classE
CAloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
ctraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul_grad/Reshape_1Reshape_training/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul_grad/Sum_1atraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul_grad/Shape_1*
T0*
Tshape0*O
_classE
CAloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul*#
_output_shapes
:���������
�
ytraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape_2_grad/ShapeShapeLloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits*
_output_shapes
:*
T0*
out_type0*i
_class_
][loc:@loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape_2
�
{training/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape_2_grad/ReshapeReshapectraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul_grad/Reshape_1ytraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape_2_grad/Shape*
T0*
Tshape0*i
_class_
][loc:@loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape_2*#
_output_shapes
:���������
�
"training/Adam/gradients/zeros_like	ZerosLikeNloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits:1*
T0*_
_classU
SQloc:@loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits*0
_output_shapes
:������������������
�
xtraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits_grad/ExpandDims/dimConst*
valueB :
���������*_
_classU
SQloc:@loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits*
dtype0*
_output_shapes
: 
�
ttraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits_grad/ExpandDims
ExpandDims{training/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape_2_grad/Reshapextraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits_grad/ExpandDims/dim*

Tdim0*
T0*_
_classU
SQloc:@loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits*'
_output_shapes
:���������
�
mtraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits_grad/mulMulttraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits_grad/ExpandDimsNloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits:1*0
_output_shapes
:������������������*
T0*_
_classU
SQloc:@loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits
�
ttraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits_grad/LogSoftmax
LogSoftmaxTloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape*0
_output_shapes
:������������������*
T0*_
_classU
SQloc:@loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits
�
mtraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits_grad/NegNegttraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits_grad/LogSoftmax*0
_output_shapes
:������������������*
T0*_
_classU
SQloc:@loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits
�
ztraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits_grad/ExpandDims_1/dimConst*
dtype0*
_output_shapes
: *
valueB :
���������*_
_classU
SQloc:@loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits
�
vtraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits_grad/ExpandDims_1
ExpandDims{training/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape_2_grad/Reshapeztraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits_grad/ExpandDims_1/dim*
T0*_
_classU
SQloc:@loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits*'
_output_shapes
:���������*

Tdim0
�
otraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits_grad/mul_1Mulvtraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits_grad/ExpandDims_1mtraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits_grad/Neg*
T0*_
_classU
SQloc:@loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits*0
_output_shapes
:������������������
�
wtraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape_grad/ShapeShapedense_5/BiasAdd*
T0*
out_type0*g
_class]
[Yloc:@loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape*
_output_shapes
:
�
ytraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape_grad/ReshapeReshapemtraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits_grad/mulwtraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape_grad/Shape*
T0*
Tshape0*g
_class]
[Yloc:@loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape*'
_output_shapes
:���������+
�
8training/Adam/gradients/dense_5/BiasAdd_grad/BiasAddGradBiasAddGradytraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape_grad/Reshape*
data_formatNHWC*
_output_shapes
:+*
T0*"
_class
loc:@dense_5/BiasAdd
�
2training/Adam/gradients/dense_5/MatMul_grad/MatMulMatMulytraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape_grad/Reshapedense_5/MatMul/ReadVariableOp*
transpose_b(*
T0*!
_class
loc:@dense_5/MatMul*(
_output_shapes
:����������*
transpose_a( 
�
4training/Adam/gradients/dense_5/MatMul_grad/MatMul_1MatMul batch_normalization_4/cond/Mergeytraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape_grad/Reshape*
T0*!
_class
loc:@dense_5/MatMul*
_output_shapes
:	�+*
transpose_a(*
transpose_b( 
�
Gtraining/Adam/gradients/batch_normalization_4/cond/Merge_grad/cond_gradSwitch2training/Adam/gradients/dense_5/MatMul_grad/MatMul"batch_normalization_4/cond/pred_id*
T0*!
_class
loc:@dense_5/MatMul*<
_output_shapes*
(:����������:����������
�
Mtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/ShapeShape*batch_normalization_4/cond/batchnorm/mul_1*
T0*
out_type0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/add_1*
_output_shapes
:
�
Otraining/Adam/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/Shape_1Shape(batch_normalization_4/cond/batchnorm/sub*
T0*
out_type0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/add_1*
_output_shapes
:
�
]training/Adam/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsMtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/ShapeOtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/add_1
�
Ktraining/Adam/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/SumSumGtraining/Adam/gradients/batch_normalization_4/cond/Merge_grad/cond_grad]training/Adam/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/BroadcastGradientArgs*
T0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/add_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
Otraining/Adam/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/ReshapeReshapeKtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/SumMtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/add_1
�
Mtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/Sum_1SumGtraining/Adam/gradients/batch_normalization_4/cond/Merge_grad/cond_grad_training/Adam/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/add_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
Qtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/Reshape_1ReshapeMtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/Sum_1Otraining/Adam/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/add_1*
_output_shapes	
:�
�
training/Adam/gradients/SwitchSwitch%batch_normalization_4/batchnorm/add_1"batch_normalization_4/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1*<
_output_shapes*
(:����������:����������
�
 training/Adam/gradients/IdentityIdentitytraining/Adam/gradients/Switch*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1*(
_output_shapes
:����������
�
training/Adam/gradients/Shape_1Shapetraining/Adam/gradients/Switch*
_output_shapes
:*
T0*
out_type0*8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1
�
#training/Adam/gradients/zeros/ConstConst!^training/Adam/gradients/Identity*
valueB
 *    *8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1*
dtype0*
_output_shapes
: 
�
training/Adam/gradients/zerosFilltraining/Adam/gradients/Shape_1#training/Adam/gradients/zeros/Const*
T0*

index_type0*8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1*(
_output_shapes
:����������
�
Jtraining/Adam/gradients/batch_normalization_4/cond/Switch_1_grad/cond_gradMergetraining/Adam/gradients/zerosItraining/Adam/gradients/batch_normalization_4/cond/Merge_grad/cond_grad:1*
N**
_output_shapes
:����������: *
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1
�
Mtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/ShapeShape1batch_normalization_4/cond/batchnorm/mul_1/Switch*
T0*
out_type0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/mul_1*
_output_shapes
:
�
Otraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/Shape_1Shape(batch_normalization_4/cond/batchnorm/mul*
_output_shapes
:*
T0*
out_type0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/mul_1
�
]training/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsMtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/ShapeOtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/Shape_1*
T0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/mul_1*2
_output_shapes 
:���������:���������
�
Ktraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/MulMulOtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/Reshape(batch_normalization_4/cond/batchnorm/mul*
T0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/mul_1*(
_output_shapes
:����������
�
Ktraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/SumSumKtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/Mul]training/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/mul_1*
_output_shapes
:
�
Otraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/ReshapeReshapeKtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/SumMtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/mul_1*(
_output_shapes
:����������
�
Mtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/Mul_1Mul1batch_normalization_4/cond/batchnorm/mul_1/SwitchOtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/Reshape*
T0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/mul_1*(
_output_shapes
:����������
�
Mtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/Sum_1SumMtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/Mul_1_training/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
T0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/mul_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
Qtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/Reshape_1ReshapeMtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/Sum_1Otraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/Shape_1*
_output_shapes	
:�*
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/mul_1
�
Itraining/Adam/gradients/batch_normalization_4/cond/batchnorm/sub_grad/NegNegQtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/Reshape_1*
T0*;
_class1
/-loc:@batch_normalization_4/cond/batchnorm/sub*
_output_shapes	
:�
�
Htraining/Adam/gradients/batch_normalization_4/batchnorm/add_1_grad/ShapeShape%batch_normalization_4/batchnorm/mul_1*
T0*
out_type0*8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1*
_output_shapes
:
�
Jtraining/Adam/gradients/batch_normalization_4/batchnorm/add_1_grad/Shape_1Shape#batch_normalization_4/batchnorm/sub*
T0*
out_type0*8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1*
_output_shapes
:
�
Xtraining/Adam/gradients/batch_normalization_4/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsHtraining/Adam/gradients/batch_normalization_4/batchnorm/add_1_grad/ShapeJtraining/Adam/gradients/batch_normalization_4/batchnorm/add_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1
�
Ftraining/Adam/gradients/batch_normalization_4/batchnorm/add_1_grad/SumSumJtraining/Adam/gradients/batch_normalization_4/cond/Switch_1_grad/cond_gradXtraining/Adam/gradients/batch_normalization_4/batchnorm/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1*
_output_shapes
:
�
Jtraining/Adam/gradients/batch_normalization_4/batchnorm/add_1_grad/ReshapeReshapeFtraining/Adam/gradients/batch_normalization_4/batchnorm/add_1_grad/SumHtraining/Adam/gradients/batch_normalization_4/batchnorm/add_1_grad/Shape*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1*(
_output_shapes
:����������
�
Htraining/Adam/gradients/batch_normalization_4/batchnorm/add_1_grad/Sum_1SumJtraining/Adam/gradients/batch_normalization_4/cond/Switch_1_grad/cond_gradZtraining/Adam/gradients/batch_normalization_4/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
Ltraining/Adam/gradients/batch_normalization_4/batchnorm/add_1_grad/Reshape_1ReshapeHtraining/Adam/gradients/batch_normalization_4/batchnorm/add_1_grad/Sum_1Jtraining/Adam/gradients/batch_normalization_4/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1*
_output_shapes	
:�
�
 training/Adam/gradients/Switch_1Switchdense_4/Relu"batch_normalization_4/cond/pred_id*
T0*
_class
loc:@dense_4/Relu*<
_output_shapes*
(:����������:����������
�
"training/Adam/gradients/Identity_1Identity"training/Adam/gradients/Switch_1:1*(
_output_shapes
:����������*
T0*
_class
loc:@dense_4/Relu
�
training/Adam/gradients/Shape_2Shape"training/Adam/gradients/Switch_1:1*
T0*
out_type0*
_class
loc:@dense_4/Relu*
_output_shapes
:
�
%training/Adam/gradients/zeros_1/ConstConst#^training/Adam/gradients/Identity_1*
valueB
 *    *
_class
loc:@dense_4/Relu*
dtype0*
_output_shapes
: 
�
training/Adam/gradients/zeros_1Filltraining/Adam/gradients/Shape_2%training/Adam/gradients/zeros_1/Const*
T0*

index_type0*
_class
loc:@dense_4/Relu*(
_output_shapes
:����������
�
Xtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_1/Switch_grad/cond_gradMergeOtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/Reshapetraining/Adam/gradients/zeros_1*
T0*
_class
loc:@dense_4/Relu*
N**
_output_shapes
:����������: 
�
Ktraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_2_grad/MulMulItraining/Adam/gradients/batch_normalization_4/cond/batchnorm/sub_grad/Neg(batch_normalization_4/cond/batchnorm/mul*
T0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/mul_2*
_output_shapes	
:�
�
Mtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_2_grad/Mul_1MulItraining/Adam/gradients/batch_normalization_4/cond/batchnorm/sub_grad/Neg5batch_normalization_4/cond/batchnorm/ReadVariableOp_1*
T0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/mul_2*
_output_shapes	
:�
�
Htraining/Adam/gradients/batch_normalization_4/batchnorm/mul_1_grad/ShapeShapedense_4/Relu*
T0*
out_type0*8
_class.
,*loc:@batch_normalization_4/batchnorm/mul_1*
_output_shapes
:
�
Jtraining/Adam/gradients/batch_normalization_4/batchnorm/mul_1_grad/Shape_1Shape#batch_normalization_4/batchnorm/mul*
_output_shapes
:*
T0*
out_type0*8
_class.
,*loc:@batch_normalization_4/batchnorm/mul_1
�
Xtraining/Adam/gradients/batch_normalization_4/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsHtraining/Adam/gradients/batch_normalization_4/batchnorm/mul_1_grad/ShapeJtraining/Adam/gradients/batch_normalization_4/batchnorm/mul_1_grad/Shape_1*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/mul_1*2
_output_shapes 
:���������:���������
�
Ftraining/Adam/gradients/batch_normalization_4/batchnorm/mul_1_grad/MulMulJtraining/Adam/gradients/batch_normalization_4/batchnorm/add_1_grad/Reshape#batch_normalization_4/batchnorm/mul*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/mul_1*(
_output_shapes
:����������
�
Ftraining/Adam/gradients/batch_normalization_4/batchnorm/mul_1_grad/SumSumFtraining/Adam/gradients/batch_normalization_4/batchnorm/mul_1_grad/MulXtraining/Adam/gradients/batch_normalization_4/batchnorm/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/mul_1
�
Jtraining/Adam/gradients/batch_normalization_4/batchnorm/mul_1_grad/ReshapeReshapeFtraining/Adam/gradients/batch_normalization_4/batchnorm/mul_1_grad/SumHtraining/Adam/gradients/batch_normalization_4/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_4/batchnorm/mul_1*(
_output_shapes
:����������
�
Htraining/Adam/gradients/batch_normalization_4/batchnorm/mul_1_grad/Mul_1Muldense_4/ReluJtraining/Adam/gradients/batch_normalization_4/batchnorm/add_1_grad/Reshape*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/mul_1*(
_output_shapes
:����������
�
Htraining/Adam/gradients/batch_normalization_4/batchnorm/mul_1_grad/Sum_1SumHtraining/Adam/gradients/batch_normalization_4/batchnorm/mul_1_grad/Mul_1Ztraining/Adam/gradients/batch_normalization_4/batchnorm/mul_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/mul_1*
_output_shapes
:
�
Ltraining/Adam/gradients/batch_normalization_4/batchnorm/mul_1_grad/Reshape_1ReshapeHtraining/Adam/gradients/batch_normalization_4/batchnorm/mul_1_grad/Sum_1Jtraining/Adam/gradients/batch_normalization_4/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_4/batchnorm/mul_1*
_output_shapes	
:�
�
Dtraining/Adam/gradients/batch_normalization_4/batchnorm/sub_grad/NegNegLtraining/Adam/gradients/batch_normalization_4/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:�*
T0*6
_class,
*(loc:@batch_normalization_4/batchnorm/sub
�
 training/Adam/gradients/Switch_2Switchbatch_normalization_4/beta"batch_normalization_4/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_4/beta*
_output_shapes
: : 
�
"training/Adam/gradients/Identity_2Identity"training/Adam/gradients/Switch_2:1*
T0*-
_class#
!loc:@batch_normalization_4/beta*
_output_shapes
: 
�
%training/Adam/gradients/VariableShapeVariableShape"training/Adam/gradients/Switch_2:1#^training/Adam/gradients/Identity_2*
_output_shapes
:*
out_type0*-
_class#
!loc:@batch_normalization_4/beta
�
%training/Adam/gradients/zeros_2/ConstConst#^training/Adam/gradients/Identity_2*
valueB
 *    *-
_class#
!loc:@batch_normalization_4/beta*
dtype0*
_output_shapes
: 
�
training/Adam/gradients/zeros_2Fill%training/Adam/gradients/VariableShape%training/Adam/gradients/zeros_2/Const*
T0*

index_type0*-
_class#
!loc:@batch_normalization_4/beta*#
_output_shapes
:���������
�
ctraining/Adam/gradients/batch_normalization_4/cond/batchnorm/ReadVariableOp_2/Switch_grad/cond_gradMergeQtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/Reshape_1training/Adam/gradients/zeros_2*
T0*-
_class#
!loc:@batch_normalization_4/beta*
N*%
_output_shapes
:���������: 
�
training/Adam/gradients/AddNAddNQtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/Reshape_1Mtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_2_grad/Mul_1*
N*
_output_shapes	
:�*
T0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/mul_1
�
Itraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_grad/MulMultraining/Adam/gradients/AddN7batch_normalization_4/cond/batchnorm/mul/ReadVariableOp*
T0*;
_class1
/-loc:@batch_normalization_4/cond/batchnorm/mul*
_output_shapes	
:�
�
Ktraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_grad/Mul_1Multraining/Adam/gradients/AddN*batch_normalization_4/cond/batchnorm/Rsqrt*
T0*;
_class1
/-loc:@batch_normalization_4/cond/batchnorm/mul*
_output_shapes	
:�
�
Ftraining/Adam/gradients/batch_normalization_4/batchnorm/mul_2_grad/MulMulDtraining/Adam/gradients/batch_normalization_4/batchnorm/sub_grad/Neg#batch_normalization_4/batchnorm/mul*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/mul_2*
_output_shapes	
:�
�
Htraining/Adam/gradients/batch_normalization_4/batchnorm/mul_2_grad/Mul_1MulDtraining/Adam/gradients/batch_normalization_4/batchnorm/sub_grad/Neg%batch_normalization_4/moments/Squeeze*
_output_shapes	
:�*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/mul_2
�
training/Adam/gradients/AddN_1AddNctraining/Adam/gradients/batch_normalization_4/cond/batchnorm/ReadVariableOp_2/Switch_grad/cond_gradLtraining/Adam/gradients/batch_normalization_4/batchnorm/add_1_grad/Reshape_1*
T0*-
_class#
!loc:@batch_normalization_4/beta*
N*
_output_shapes	
:�
�
Htraining/Adam/gradients/batch_normalization_4/moments/Squeeze_grad/ShapeConst*
valueB"   �   *8
_class.
,*loc:@batch_normalization_4/moments/Squeeze*
dtype0*
_output_shapes
:
�
Jtraining/Adam/gradients/batch_normalization_4/moments/Squeeze_grad/ReshapeReshapeFtraining/Adam/gradients/batch_normalization_4/batchnorm/mul_2_grad/MulHtraining/Adam/gradients/batch_normalization_4/moments/Squeeze_grad/Shape*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_4/moments/Squeeze*
_output_shapes
:	�
�
training/Adam/gradients/AddN_2AddNLtraining/Adam/gradients/batch_normalization_4/batchnorm/mul_1_grad/Reshape_1Htraining/Adam/gradients/batch_normalization_4/batchnorm/mul_2_grad/Mul_1*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/mul_1*
N*
_output_shapes	
:�
�
Dtraining/Adam/gradients/batch_normalization_4/batchnorm/mul_grad/MulMultraining/Adam/gradients/AddN_22batch_normalization_4/batchnorm/mul/ReadVariableOp*
_output_shapes	
:�*
T0*6
_class,
*(loc:@batch_normalization_4/batchnorm/mul
�
Ftraining/Adam/gradients/batch_normalization_4/batchnorm/mul_grad/Mul_1Multraining/Adam/gradients/AddN_2%batch_normalization_4/batchnorm/Rsqrt*
T0*6
_class,
*(loc:@batch_normalization_4/batchnorm/mul*
_output_shapes	
:�
�
 training/Adam/gradients/Switch_3Switchbatch_normalization_4/gamma"batch_normalization_4/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_4/gamma*
_output_shapes
: : 
�
"training/Adam/gradients/Identity_3Identity"training/Adam/gradients/Switch_3:1*
_output_shapes
: *
T0*.
_class$
" loc:@batch_normalization_4/gamma
�
'training/Adam/gradients/VariableShape_1VariableShape"training/Adam/gradients/Switch_3:1#^training/Adam/gradients/Identity_3*
out_type0*.
_class$
" loc:@batch_normalization_4/gamma*
_output_shapes
:
�
%training/Adam/gradients/zeros_3/ConstConst#^training/Adam/gradients/Identity_3*
valueB
 *    *.
_class$
" loc:@batch_normalization_4/gamma*
dtype0*
_output_shapes
: 
�
training/Adam/gradients/zeros_3Fill'training/Adam/gradients/VariableShape_1%training/Adam/gradients/zeros_3/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization_4/gamma*#
_output_shapes
:���������
�
etraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul/ReadVariableOp/Switch_grad/cond_gradMergeKtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_grad/Mul_1training/Adam/gradients/zeros_3*
N*%
_output_shapes
:���������: *
T0*.
_class$
" loc:@batch_normalization_4/gamma
�
Ltraining/Adam/gradients/batch_normalization_4/batchnorm/Rsqrt_grad/RsqrtGrad	RsqrtGrad%batch_normalization_4/batchnorm/RsqrtDtraining/Adam/gradients/batch_normalization_4/batchnorm/mul_grad/Mul*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/Rsqrt*
_output_shapes	
:�
�
Ytraining/Adam/gradients/batch_normalization_4/batchnorm/add_grad/BroadcastGradientArgs/s0Const*
valueB:�*6
_class,
*(loc:@batch_normalization_4/batchnorm/add*
dtype0*
_output_shapes
:
�
Ytraining/Adam/gradients/batch_normalization_4/batchnorm/add_grad/BroadcastGradientArgs/s1Const*
valueB *6
_class,
*(loc:@batch_normalization_4/batchnorm/add*
dtype0*
_output_shapes
: 
�
Vtraining/Adam/gradients/batch_normalization_4/batchnorm/add_grad/BroadcastGradientArgsBroadcastGradientArgsYtraining/Adam/gradients/batch_normalization_4/batchnorm/add_grad/BroadcastGradientArgs/s0Ytraining/Adam/gradients/batch_normalization_4/batchnorm/add_grad/BroadcastGradientArgs/s1*2
_output_shapes 
:���������:���������*
T0*6
_class,
*(loc:@batch_normalization_4/batchnorm/add
�
Vtraining/Adam/gradients/batch_normalization_4/batchnorm/add_grad/Sum/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB: *6
_class,
*(loc:@batch_normalization_4/batchnorm/add
�
Dtraining/Adam/gradients/batch_normalization_4/batchnorm/add_grad/SumSumLtraining/Adam/gradients/batch_normalization_4/batchnorm/Rsqrt_grad/RsqrtGradVtraining/Adam/gradients/batch_normalization_4/batchnorm/add_grad/Sum/reduction_indices*
T0*6
_class,
*(loc:@batch_normalization_4/batchnorm/add*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
Ntraining/Adam/gradients/batch_normalization_4/batchnorm/add_grad/Reshape/shapeConst*
dtype0*
_output_shapes
: *
valueB *6
_class,
*(loc:@batch_normalization_4/batchnorm/add
�
Htraining/Adam/gradients/batch_normalization_4/batchnorm/add_grad/ReshapeReshapeDtraining/Adam/gradients/batch_normalization_4/batchnorm/add_grad/SumNtraining/Adam/gradients/batch_normalization_4/batchnorm/add_grad/Reshape/shape*
T0*
Tshape0*6
_class,
*(loc:@batch_normalization_4/batchnorm/add*
_output_shapes
: 
�
training/Adam/gradients/AddN_3AddNetraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul/ReadVariableOp/Switch_grad/cond_gradFtraining/Adam/gradients/batch_normalization_4/batchnorm/mul_grad/Mul_1*
N*
_output_shapes	
:�*
T0*.
_class$
" loc:@batch_normalization_4/gamma
�
Jtraining/Adam/gradients/batch_normalization_4/moments/Squeeze_1_grad/ShapeConst*
valueB"   �   *:
_class0
.,loc:@batch_normalization_4/moments/Squeeze_1*
dtype0*
_output_shapes
:
�
Ltraining/Adam/gradients/batch_normalization_4/moments/Squeeze_1_grad/ReshapeReshapeLtraining/Adam/gradients/batch_normalization_4/batchnorm/Rsqrt_grad/RsqrtGradJtraining/Adam/gradients/batch_normalization_4/moments/Squeeze_1_grad/Shape*
T0*
Tshape0*:
_class0
.,loc:@batch_normalization_4/moments/Squeeze_1*
_output_shapes
:	�
�
Itraining/Adam/gradients/batch_normalization_4/moments/variance_grad/ShapeShape/batch_normalization_4/moments/SquaredDifference*
T0*
out_type0*9
_class/
-+loc:@batch_normalization_4/moments/variance*
_output_shapes
:
�
Htraining/Adam/gradients/batch_normalization_4/moments/variance_grad/SizeConst*
value	B :*9
_class/
-+loc:@batch_normalization_4/moments/variance*
dtype0*
_output_shapes
: 
�
Gtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/addAddV28batch_normalization_4/moments/variance/reduction_indicesHtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/Size*
_output_shapes
:*
T0*9
_class/
-+loc:@batch_normalization_4/moments/variance
�
Gtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/modFloorModGtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/addHtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/Size*
_output_shapes
:*
T0*9
_class/
-+loc:@batch_normalization_4/moments/variance
�
Ktraining/Adam/gradients/batch_normalization_4/moments/variance_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:*9
_class/
-+loc:@batch_normalization_4/moments/variance
�
Otraining/Adam/gradients/batch_normalization_4/moments/variance_grad/range/startConst*
dtype0*
_output_shapes
: *
value	B : *9
_class/
-+loc:@batch_normalization_4/moments/variance
�
Otraining/Adam/gradients/batch_normalization_4/moments/variance_grad/range/deltaConst*
value	B :*9
_class/
-+loc:@batch_normalization_4/moments/variance*
dtype0*
_output_shapes
: 
�
Itraining/Adam/gradients/batch_normalization_4/moments/variance_grad/rangeRangeOtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/range/startHtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/SizeOtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/range/delta*9
_class/
-+loc:@batch_normalization_4/moments/variance*
_output_shapes
:*

Tidx0
�
Ntraining/Adam/gradients/batch_normalization_4/moments/variance_grad/Fill/valueConst*
value	B :*9
_class/
-+loc:@batch_normalization_4/moments/variance*
dtype0*
_output_shapes
: 
�
Htraining/Adam/gradients/batch_normalization_4/moments/variance_grad/FillFillKtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/Shape_1Ntraining/Adam/gradients/batch_normalization_4/moments/variance_grad/Fill/value*
T0*

index_type0*9
_class/
-+loc:@batch_normalization_4/moments/variance*
_output_shapes
:
�
Qtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/DynamicStitchDynamicStitchItraining/Adam/gradients/batch_normalization_4/moments/variance_grad/rangeGtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/modItraining/Adam/gradients/batch_normalization_4/moments/variance_grad/ShapeHtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/Fill*
T0*9
_class/
-+loc:@batch_normalization_4/moments/variance*
N*
_output_shapes
:
�
Mtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/Maximum/yConst*
value	B :*9
_class/
-+loc:@batch_normalization_4/moments/variance*
dtype0*
_output_shapes
: 
�
Ktraining/Adam/gradients/batch_normalization_4/moments/variance_grad/MaximumMaximumQtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/DynamicStitchMtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/Maximum/y*
T0*9
_class/
-+loc:@batch_normalization_4/moments/variance*
_output_shapes
:
�
Ltraining/Adam/gradients/batch_normalization_4/moments/variance_grad/floordivFloorDivItraining/Adam/gradients/batch_normalization_4/moments/variance_grad/ShapeKtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/Maximum*
T0*9
_class/
-+loc:@batch_normalization_4/moments/variance*
_output_shapes
:
�
Ktraining/Adam/gradients/batch_normalization_4/moments/variance_grad/ReshapeReshapeLtraining/Adam/gradients/batch_normalization_4/moments/Squeeze_1_grad/ReshapeQtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/DynamicStitch*
T0*
Tshape0*9
_class/
-+loc:@batch_normalization_4/moments/variance*0
_output_shapes
:������������������
�
Htraining/Adam/gradients/batch_normalization_4/moments/variance_grad/TileTileKtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/ReshapeLtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/floordiv*

Tmultiples0*
T0*9
_class/
-+loc:@batch_normalization_4/moments/variance*0
_output_shapes
:������������������
�
Ktraining/Adam/gradients/batch_normalization_4/moments/variance_grad/Shape_2Shape/batch_normalization_4/moments/SquaredDifference*
_output_shapes
:*
T0*
out_type0*9
_class/
-+loc:@batch_normalization_4/moments/variance
�
Ktraining/Adam/gradients/batch_normalization_4/moments/variance_grad/Shape_3Const*
valueB"   �   *9
_class/
-+loc:@batch_normalization_4/moments/variance*
dtype0*
_output_shapes
:
�
Itraining/Adam/gradients/batch_normalization_4/moments/variance_grad/ConstConst*
valueB: *9
_class/
-+loc:@batch_normalization_4/moments/variance*
dtype0*
_output_shapes
:
�
Htraining/Adam/gradients/batch_normalization_4/moments/variance_grad/ProdProdKtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/Shape_2Itraining/Adam/gradients/batch_normalization_4/moments/variance_grad/Const*
T0*9
_class/
-+loc:@batch_normalization_4/moments/variance*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
Ktraining/Adam/gradients/batch_normalization_4/moments/variance_grad/Const_1Const*
valueB: *9
_class/
-+loc:@batch_normalization_4/moments/variance*
dtype0*
_output_shapes
:
�
Jtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/Prod_1ProdKtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/Shape_3Ktraining/Adam/gradients/batch_normalization_4/moments/variance_grad/Const_1*
T0*9
_class/
-+loc:@batch_normalization_4/moments/variance*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
Otraining/Adam/gradients/batch_normalization_4/moments/variance_grad/Maximum_1/yConst*
dtype0*
_output_shapes
: *
value	B :*9
_class/
-+loc:@batch_normalization_4/moments/variance
�
Mtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/Maximum_1MaximumJtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/Prod_1Otraining/Adam/gradients/batch_normalization_4/moments/variance_grad/Maximum_1/y*
T0*9
_class/
-+loc:@batch_normalization_4/moments/variance*
_output_shapes
: 
�
Ntraining/Adam/gradients/batch_normalization_4/moments/variance_grad/floordiv_1FloorDivHtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/ProdMtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/Maximum_1*
_output_shapes
: *
T0*9
_class/
-+loc:@batch_normalization_4/moments/variance
�
Htraining/Adam/gradients/batch_normalization_4/moments/variance_grad/CastCastNtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/floordiv_1*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0*9
_class/
-+loc:@batch_normalization_4/moments/variance
�
Ktraining/Adam/gradients/batch_normalization_4/moments/variance_grad/truedivRealDivHtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/TileHtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/Cast*
T0*9
_class/
-+loc:@batch_normalization_4/moments/variance*(
_output_shapes
:����������
�
Straining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/scalarConstL^training/Adam/gradients/batch_normalization_4/moments/variance_grad/truediv*
valueB
 *   @*B
_class8
64loc:@batch_normalization_4/moments/SquaredDifference*
dtype0*
_output_shapes
: 
�
Ptraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/MulMulStraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/scalarKtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/truediv*
T0*B
_class8
64loc:@batch_normalization_4/moments/SquaredDifference*(
_output_shapes
:����������
�
Ptraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/subSubdense_4/Relu*batch_normalization_4/moments/StopGradientL^training/Adam/gradients/batch_normalization_4/moments/variance_grad/truediv*
T0*B
_class8
64loc:@batch_normalization_4/moments/SquaredDifference*(
_output_shapes
:����������
�
Rtraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/mul_1MulPtraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/MulPtraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/sub*
T0*B
_class8
64loc:@batch_normalization_4/moments/SquaredDifference*(
_output_shapes
:����������
�
Rtraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/ShapeShapedense_4/Relu*
T0*
out_type0*B
_class8
64loc:@batch_normalization_4/moments/SquaredDifference*
_output_shapes
:
�
Ttraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/Shape_1Shape*batch_normalization_4/moments/StopGradient*
_output_shapes
:*
T0*
out_type0*B
_class8
64loc:@batch_normalization_4/moments/SquaredDifference
�
btraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgsRtraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/ShapeTtraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/Shape_1*
T0*B
_class8
64loc:@batch_normalization_4/moments/SquaredDifference*2
_output_shapes 
:���������:���������
�
Ptraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/SumSumRtraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/mul_1btraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*B
_class8
64loc:@batch_normalization_4/moments/SquaredDifference
�
Ttraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/ReshapeReshapePtraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/SumRtraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/Shape*
T0*
Tshape0*B
_class8
64loc:@batch_normalization_4/moments/SquaredDifference*(
_output_shapes
:����������
�
Rtraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/Sum_1SumRtraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/mul_1dtraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*B
_class8
64loc:@batch_normalization_4/moments/SquaredDifference*
_output_shapes
:
�
Vtraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/Reshape_1ReshapeRtraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/Sum_1Ttraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/Shape_1*
T0*
Tshape0*B
_class8
64loc:@batch_normalization_4/moments/SquaredDifference*
_output_shapes
:	�
�
Ptraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/NegNegVtraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/Reshape_1*
T0*B
_class8
64loc:@batch_normalization_4/moments/SquaredDifference*
_output_shapes
:	�
�
Etraining/Adam/gradients/batch_normalization_4/moments/mean_grad/ShapeShapedense_4/Relu*
T0*
out_type0*5
_class+
)'loc:@batch_normalization_4/moments/mean*
_output_shapes
:
�
Dtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/SizeConst*
value	B :*5
_class+
)'loc:@batch_normalization_4/moments/mean*
dtype0*
_output_shapes
: 
�
Ctraining/Adam/gradients/batch_normalization_4/moments/mean_grad/addAddV24batch_normalization_4/moments/mean/reduction_indicesDtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Size*
T0*5
_class+
)'loc:@batch_normalization_4/moments/mean*
_output_shapes
:
�
Ctraining/Adam/gradients/batch_normalization_4/moments/mean_grad/modFloorModCtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/addDtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Size*
T0*5
_class+
)'loc:@batch_normalization_4/moments/mean*
_output_shapes
:
�
Gtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Shape_1Const*
valueB:*5
_class+
)'loc:@batch_normalization_4/moments/mean*
dtype0*
_output_shapes
:
�
Ktraining/Adam/gradients/batch_normalization_4/moments/mean_grad/range/startConst*
dtype0*
_output_shapes
: *
value	B : *5
_class+
)'loc:@batch_normalization_4/moments/mean
�
Ktraining/Adam/gradients/batch_normalization_4/moments/mean_grad/range/deltaConst*
value	B :*5
_class+
)'loc:@batch_normalization_4/moments/mean*
dtype0*
_output_shapes
: 
�
Etraining/Adam/gradients/batch_normalization_4/moments/mean_grad/rangeRangeKtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/range/startDtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/SizeKtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/range/delta*5
_class+
)'loc:@batch_normalization_4/moments/mean*
_output_shapes
:*

Tidx0
�
Jtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Fill/valueConst*
value	B :*5
_class+
)'loc:@batch_normalization_4/moments/mean*
dtype0*
_output_shapes
: 
�
Dtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/FillFillGtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Shape_1Jtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Fill/value*
T0*

index_type0*5
_class+
)'loc:@batch_normalization_4/moments/mean*
_output_shapes
:
�
Mtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/DynamicStitchDynamicStitchEtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/rangeCtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/modEtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/ShapeDtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Fill*
T0*5
_class+
)'loc:@batch_normalization_4/moments/mean*
N*
_output_shapes
:
�
Itraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Maximum/yConst*
value	B :*5
_class+
)'loc:@batch_normalization_4/moments/mean*
dtype0*
_output_shapes
: 
�
Gtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/MaximumMaximumMtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/DynamicStitchItraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Maximum/y*
T0*5
_class+
)'loc:@batch_normalization_4/moments/mean*
_output_shapes
:
�
Htraining/Adam/gradients/batch_normalization_4/moments/mean_grad/floordivFloorDivEtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/ShapeGtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Maximum*
_output_shapes
:*
T0*5
_class+
)'loc:@batch_normalization_4/moments/mean
�
Gtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/ReshapeReshapeJtraining/Adam/gradients/batch_normalization_4/moments/Squeeze_grad/ReshapeMtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/DynamicStitch*
T0*
Tshape0*5
_class+
)'loc:@batch_normalization_4/moments/mean*0
_output_shapes
:������������������
�
Dtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/TileTileGtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/ReshapeHtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/floordiv*

Tmultiples0*
T0*5
_class+
)'loc:@batch_normalization_4/moments/mean*0
_output_shapes
:������������������
�
Gtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Shape_2Shapedense_4/Relu*
_output_shapes
:*
T0*
out_type0*5
_class+
)'loc:@batch_normalization_4/moments/mean
�
Gtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Shape_3Const*
valueB"   �   *5
_class+
)'loc:@batch_normalization_4/moments/mean*
dtype0*
_output_shapes
:
�
Etraining/Adam/gradients/batch_normalization_4/moments/mean_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: *5
_class+
)'loc:@batch_normalization_4/moments/mean
�
Dtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/ProdProdGtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Shape_2Etraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Const*
T0*5
_class+
)'loc:@batch_normalization_4/moments/mean*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
Gtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Const_1Const*
valueB: *5
_class+
)'loc:@batch_normalization_4/moments/mean*
dtype0*
_output_shapes
:
�
Ftraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Prod_1ProdGtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Shape_3Gtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Const_1*
T0*5
_class+
)'loc:@batch_normalization_4/moments/mean*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
Ktraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Maximum_1/yConst*
value	B :*5
_class+
)'loc:@batch_normalization_4/moments/mean*
dtype0*
_output_shapes
: 
�
Itraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Maximum_1MaximumFtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Prod_1Ktraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Maximum_1/y*
T0*5
_class+
)'loc:@batch_normalization_4/moments/mean*
_output_shapes
: 
�
Jtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/floordiv_1FloorDivDtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/ProdItraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Maximum_1*
T0*5
_class+
)'loc:@batch_normalization_4/moments/mean*
_output_shapes
: 
�
Dtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/CastCastJtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/floordiv_1*

SrcT0*5
_class+
)'loc:@batch_normalization_4/moments/mean*
Truncate( *
_output_shapes
: *

DstT0
�
Gtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/truedivRealDivDtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/TileDtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Cast*
T0*5
_class+
)'loc:@batch_normalization_4/moments/mean*(
_output_shapes
:����������
�
training/Adam/gradients/AddN_4AddNXtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_1/Switch_grad/cond_gradJtraining/Adam/gradients/batch_normalization_4/batchnorm/mul_1_grad/ReshapeTtraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/ReshapeGtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/truediv*
T0*
_class
loc:@dense_4/Relu*
N*(
_output_shapes
:����������
�
2training/Adam/gradients/dense_4/Relu_grad/ReluGradReluGradtraining/Adam/gradients/AddN_4dense_4/Relu*
T0*
_class
loc:@dense_4/Relu*(
_output_shapes
:����������
�
8training/Adam/gradients/dense_4/BiasAdd_grad/BiasAddGradBiasAddGrad2training/Adam/gradients/dense_4/Relu_grad/ReluGrad*
T0*"
_class
loc:@dense_4/BiasAdd*
data_formatNHWC*
_output_shapes	
:�
�
2training/Adam/gradients/dense_4/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_4/Relu_grad/ReluGraddense_4/MatMul/ReadVariableOp*
T0*!
_class
loc:@dense_4/MatMul*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
4training/Adam/gradients/dense_4/MatMul_grad/MatMul_1MatMuldropout_2/cond/Merge2training/Adam/gradients/dense_4/Relu_grad/ReluGrad*
transpose_b( *
T0*!
_class
loc:@dense_4/MatMul* 
_output_shapes
:
��*
transpose_a(
�
;training/Adam/gradients/dropout_2/cond/Merge_grad/cond_gradSwitch2training/Adam/gradients/dense_4/MatMul_grad/MatMuldropout_2/cond/pred_id*<
_output_shapes*
(:����������:����������*
T0*!
_class
loc:@dense_4/MatMul
�
 training/Adam/gradients/Switch_4Switch batch_normalization_3/cond/Mergedropout_2/cond/pred_id*
T0*3
_class)
'%loc:@batch_normalization_3/cond/Merge*<
_output_shapes*
(:����������:����������
�
"training/Adam/gradients/Identity_4Identity"training/Adam/gradients/Switch_4:1*(
_output_shapes
:����������*
T0*3
_class)
'%loc:@batch_normalization_3/cond/Merge
�
training/Adam/gradients/Shape_3Shape"training/Adam/gradients/Switch_4:1*
T0*
out_type0*3
_class)
'%loc:@batch_normalization_3/cond/Merge*
_output_shapes
:
�
%training/Adam/gradients/zeros_4/ConstConst#^training/Adam/gradients/Identity_4*
valueB
 *    *3
_class)
'%loc:@batch_normalization_3/cond/Merge*
dtype0*
_output_shapes
: 
�
training/Adam/gradients/zeros_4Filltraining/Adam/gradients/Shape_3%training/Adam/gradients/zeros_4/Const*
T0*

index_type0*3
_class)
'%loc:@batch_normalization_3/cond/Merge*(
_output_shapes
:����������
�
>training/Adam/gradients/dropout_2/cond/Switch_1_grad/cond_gradMerge;training/Adam/gradients/dropout_2/cond/Merge_grad/cond_gradtraining/Adam/gradients/zeros_4*
T0*3
_class)
'%loc:@batch_normalization_3/cond/Merge*
N**
_output_shapes
:����������: 
�
?training/Adam/gradients/dropout_2/cond/dropout/mul_1_grad/ShapeShapedropout_2/cond/dropout/mul*
T0*
out_type0*/
_class%
#!loc:@dropout_2/cond/dropout/mul_1*
_output_shapes
:
�
Atraining/Adam/gradients/dropout_2/cond/dropout/mul_1_grad/Shape_1Shapedropout_2/cond/dropout/Cast*
T0*
out_type0*/
_class%
#!loc:@dropout_2/cond/dropout/mul_1*
_output_shapes
:
�
Otraining/Adam/gradients/dropout_2/cond/dropout/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs?training/Adam/gradients/dropout_2/cond/dropout/mul_1_grad/ShapeAtraining/Adam/gradients/dropout_2/cond/dropout/mul_1_grad/Shape_1*
T0*/
_class%
#!loc:@dropout_2/cond/dropout/mul_1*2
_output_shapes 
:���������:���������
�
=training/Adam/gradients/dropout_2/cond/dropout/mul_1_grad/MulMul=training/Adam/gradients/dropout_2/cond/Merge_grad/cond_grad:1dropout_2/cond/dropout/Cast*(
_output_shapes
:����������*
T0*/
_class%
#!loc:@dropout_2/cond/dropout/mul_1
�
=training/Adam/gradients/dropout_2/cond/dropout/mul_1_grad/SumSum=training/Adam/gradients/dropout_2/cond/dropout/mul_1_grad/MulOtraining/Adam/gradients/dropout_2/cond/dropout/mul_1_grad/BroadcastGradientArgs*
T0*/
_class%
#!loc:@dropout_2/cond/dropout/mul_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
Atraining/Adam/gradients/dropout_2/cond/dropout/mul_1_grad/ReshapeReshape=training/Adam/gradients/dropout_2/cond/dropout/mul_1_grad/Sum?training/Adam/gradients/dropout_2/cond/dropout/mul_1_grad/Shape*
T0*
Tshape0*/
_class%
#!loc:@dropout_2/cond/dropout/mul_1*(
_output_shapes
:����������
�
?training/Adam/gradients/dropout_2/cond/dropout/mul_1_grad/Mul_1Muldropout_2/cond/dropout/mul=training/Adam/gradients/dropout_2/cond/Merge_grad/cond_grad:1*
T0*/
_class%
#!loc:@dropout_2/cond/dropout/mul_1*(
_output_shapes
:����������
�
?training/Adam/gradients/dropout_2/cond/dropout/mul_1_grad/Sum_1Sum?training/Adam/gradients/dropout_2/cond/dropout/mul_1_grad/Mul_1Qtraining/Adam/gradients/dropout_2/cond/dropout/mul_1_grad/BroadcastGradientArgs:1*
T0*/
_class%
#!loc:@dropout_2/cond/dropout/mul_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
Ctraining/Adam/gradients/dropout_2/cond/dropout/mul_1_grad/Reshape_1Reshape?training/Adam/gradients/dropout_2/cond/dropout/mul_1_grad/Sum_1Atraining/Adam/gradients/dropout_2/cond/dropout/mul_1_grad/Shape_1*
T0*
Tshape0*/
_class%
#!loc:@dropout_2/cond/dropout/mul_1*(
_output_shapes
:����������
�
=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/ShapeShape%dropout_2/cond/dropout/Shape/Switch:1*
_output_shapes
:*
T0*
out_type0*-
_class#
!loc:@dropout_2/cond/dropout/mul
�
?training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Shape_1Shapedropout_2/cond/dropout/truediv*
T0*
out_type0*-
_class#
!loc:@dropout_2/cond/dropout/mul*
_output_shapes
: 
�
Mtraining/Adam/gradients/dropout_2/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Shape?training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Shape_1*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul*2
_output_shapes 
:���������:���������
�
;training/Adam/gradients/dropout_2/cond/dropout/mul_grad/MulMulAtraining/Adam/gradients/dropout_2/cond/dropout/mul_1_grad/Reshapedropout_2/cond/dropout/truediv*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul*(
_output_shapes
:����������
�
;training/Adam/gradients/dropout_2/cond/dropout/mul_grad/SumSum;training/Adam/gradients/dropout_2/cond/dropout/mul_grad/MulMtraining/Adam/gradients/dropout_2/cond/dropout/mul_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul
�
?training/Adam/gradients/dropout_2/cond/dropout/mul_grad/ReshapeReshape;training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Sum=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Shape*
T0*
Tshape0*-
_class#
!loc:@dropout_2/cond/dropout/mul*(
_output_shapes
:����������
�
=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Mul_1Mul%dropout_2/cond/dropout/Shape/Switch:1Atraining/Adam/gradients/dropout_2/cond/dropout/mul_1_grad/Reshape*(
_output_shapes
:����������*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul
�
=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Sum_1Sum=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Mul_1Otraining/Adam/gradients/dropout_2/cond/dropout/mul_grad/BroadcastGradientArgs:1*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
Atraining/Adam/gradients/dropout_2/cond/dropout/mul_grad/Reshape_1Reshape=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Sum_1?training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0*-
_class#
!loc:@dropout_2/cond/dropout/mul
�
 training/Adam/gradients/Switch_5Switch batch_normalization_3/cond/Mergedropout_2/cond/pred_id*
T0*3
_class)
'%loc:@batch_normalization_3/cond/Merge*<
_output_shapes*
(:����������:����������
�
"training/Adam/gradients/Identity_5Identity training/Adam/gradients/Switch_5*(
_output_shapes
:����������*
T0*3
_class)
'%loc:@batch_normalization_3/cond/Merge
�
training/Adam/gradients/Shape_4Shape training/Adam/gradients/Switch_5*
T0*
out_type0*3
_class)
'%loc:@batch_normalization_3/cond/Merge*
_output_shapes
:
�
%training/Adam/gradients/zeros_5/ConstConst#^training/Adam/gradients/Identity_5*
valueB
 *    *3
_class)
'%loc:@batch_normalization_3/cond/Merge*
dtype0*
_output_shapes
: 
�
training/Adam/gradients/zeros_5Filltraining/Adam/gradients/Shape_4%training/Adam/gradients/zeros_5/Const*(
_output_shapes
:����������*
T0*

index_type0*3
_class)
'%loc:@batch_normalization_3/cond/Merge
�
Jtraining/Adam/gradients/dropout_2/cond/dropout/Shape/Switch_grad/cond_gradMergetraining/Adam/gradients/zeros_5?training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Reshape*
T0*3
_class)
'%loc:@batch_normalization_3/cond/Merge*
N**
_output_shapes
:����������: 
�
training/Adam/gradients/AddN_5AddN>training/Adam/gradients/dropout_2/cond/Switch_1_grad/cond_gradJtraining/Adam/gradients/dropout_2/cond/dropout/Shape/Switch_grad/cond_grad*
T0*3
_class)
'%loc:@batch_normalization_3/cond/Merge*
N*(
_output_shapes
:����������
�
Gtraining/Adam/gradients/batch_normalization_3/cond/Merge_grad/cond_gradSwitchtraining/Adam/gradients/AddN_5"batch_normalization_3/cond/pred_id*
T0*3
_class)
'%loc:@batch_normalization_3/cond/Merge*<
_output_shapes*
(:����������:����������
�
Mtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/ShapeShape*batch_normalization_3/cond/batchnorm/mul_1*
T0*
out_type0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/add_1*
_output_shapes
:
�
Otraining/Adam/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/Shape_1Shape(batch_normalization_3/cond/batchnorm/sub*
T0*
out_type0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/add_1*
_output_shapes
:
�
]training/Adam/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsMtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/ShapeOtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/add_1
�
Ktraining/Adam/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/SumSumGtraining/Adam/gradients/batch_normalization_3/cond/Merge_grad/cond_grad]training/Adam/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/add_1*
_output_shapes
:
�
Otraining/Adam/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/ReshapeReshapeKtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/SumMtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/Shape*
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/add_1*(
_output_shapes
:����������
�
Mtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/Sum_1SumGtraining/Adam/gradients/batch_normalization_3/cond/Merge_grad/cond_grad_training/Adam/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/add_1
�
Qtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/Reshape_1ReshapeMtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/Sum_1Otraining/Adam/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/add_1*
_output_shapes	
:�
�
 training/Adam/gradients/Switch_6Switch%batch_normalization_3/batchnorm/add_1"batch_normalization_3/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1*<
_output_shapes*
(:����������:����������
�
"training/Adam/gradients/Identity_6Identity training/Adam/gradients/Switch_6*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1*(
_output_shapes
:����������
�
training/Adam/gradients/Shape_5Shape training/Adam/gradients/Switch_6*
T0*
out_type0*8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1*
_output_shapes
:
�
%training/Adam/gradients/zeros_6/ConstConst#^training/Adam/gradients/Identity_6*
valueB
 *    *8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1*
dtype0*
_output_shapes
: 
�
training/Adam/gradients/zeros_6Filltraining/Adam/gradients/Shape_5%training/Adam/gradients/zeros_6/Const*
T0*

index_type0*8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1*(
_output_shapes
:����������
�
Jtraining/Adam/gradients/batch_normalization_3/cond/Switch_1_grad/cond_gradMergetraining/Adam/gradients/zeros_6Itraining/Adam/gradients/batch_normalization_3/cond/Merge_grad/cond_grad:1*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1*
N**
_output_shapes
:����������: 
�
Mtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/ShapeShape1batch_normalization_3/cond/batchnorm/mul_1/Switch*
T0*
out_type0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/mul_1*
_output_shapes
:
�
Otraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/Shape_1Shape(batch_normalization_3/cond/batchnorm/mul*
_output_shapes
:*
T0*
out_type0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/mul_1
�
]training/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsMtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/ShapeOtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/Shape_1*
T0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/mul_1*2
_output_shapes 
:���������:���������
�
Ktraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/MulMulOtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/Reshape(batch_normalization_3/cond/batchnorm/mul*
T0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/mul_1*(
_output_shapes
:����������
�
Ktraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/SumSumKtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/Mul]training/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/BroadcastGradientArgs*
T0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/mul_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
Otraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/ReshapeReshapeKtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/SumMtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/mul_1
�
Mtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/Mul_1Mul1batch_normalization_3/cond/batchnorm/mul_1/SwitchOtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/Reshape*
T0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/mul_1*(
_output_shapes
:����������
�
Mtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/Sum_1SumMtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/Mul_1_training/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
T0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/mul_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
Qtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/Reshape_1ReshapeMtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/Sum_1Otraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/mul_1*
_output_shapes	
:�
�
Itraining/Adam/gradients/batch_normalization_3/cond/batchnorm/sub_grad/NegNegQtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/Reshape_1*
T0*;
_class1
/-loc:@batch_normalization_3/cond/batchnorm/sub*
_output_shapes	
:�
�
Htraining/Adam/gradients/batch_normalization_3/batchnorm/add_1_grad/ShapeShape%batch_normalization_3/batchnorm/mul_1*
T0*
out_type0*8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1*
_output_shapes
:
�
Jtraining/Adam/gradients/batch_normalization_3/batchnorm/add_1_grad/Shape_1Shape#batch_normalization_3/batchnorm/sub*
T0*
out_type0*8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1*
_output_shapes
:
�
Xtraining/Adam/gradients/batch_normalization_3/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsHtraining/Adam/gradients/batch_normalization_3/batchnorm/add_1_grad/ShapeJtraining/Adam/gradients/batch_normalization_3/batchnorm/add_1_grad/Shape_1*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1*2
_output_shapes 
:���������:���������
�
Ftraining/Adam/gradients/batch_normalization_3/batchnorm/add_1_grad/SumSumJtraining/Adam/gradients/batch_normalization_3/cond/Switch_1_grad/cond_gradXtraining/Adam/gradients/batch_normalization_3/batchnorm/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1*
_output_shapes
:
�
Jtraining/Adam/gradients/batch_normalization_3/batchnorm/add_1_grad/ReshapeReshapeFtraining/Adam/gradients/batch_normalization_3/batchnorm/add_1_grad/SumHtraining/Adam/gradients/batch_normalization_3/batchnorm/add_1_grad/Shape*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1*(
_output_shapes
:����������
�
Htraining/Adam/gradients/batch_normalization_3/batchnorm/add_1_grad/Sum_1SumJtraining/Adam/gradients/batch_normalization_3/cond/Switch_1_grad/cond_gradZtraining/Adam/gradients/batch_normalization_3/batchnorm/add_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1*
_output_shapes
:
�
Ltraining/Adam/gradients/batch_normalization_3/batchnorm/add_1_grad/Reshape_1ReshapeHtraining/Adam/gradients/batch_normalization_3/batchnorm/add_1_grad/Sum_1Jtraining/Adam/gradients/batch_normalization_3/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1*
_output_shapes	
:�
�
 training/Adam/gradients/Switch_7Switchdense_3/Relu"batch_normalization_3/cond/pred_id*<
_output_shapes*
(:����������:����������*
T0*
_class
loc:@dense_3/Relu
�
"training/Adam/gradients/Identity_7Identity"training/Adam/gradients/Switch_7:1*(
_output_shapes
:����������*
T0*
_class
loc:@dense_3/Relu
�
training/Adam/gradients/Shape_6Shape"training/Adam/gradients/Switch_7:1*
_output_shapes
:*
T0*
out_type0*
_class
loc:@dense_3/Relu
�
%training/Adam/gradients/zeros_7/ConstConst#^training/Adam/gradients/Identity_7*
valueB
 *    *
_class
loc:@dense_3/Relu*
dtype0*
_output_shapes
: 
�
training/Adam/gradients/zeros_7Filltraining/Adam/gradients/Shape_6%training/Adam/gradients/zeros_7/Const*(
_output_shapes
:����������*
T0*

index_type0*
_class
loc:@dense_3/Relu
�
Xtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_1/Switch_grad/cond_gradMergeOtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/Reshapetraining/Adam/gradients/zeros_7*
T0*
_class
loc:@dense_3/Relu*
N**
_output_shapes
:����������: 
�
Ktraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_2_grad/MulMulItraining/Adam/gradients/batch_normalization_3/cond/batchnorm/sub_grad/Neg(batch_normalization_3/cond/batchnorm/mul*
T0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/mul_2*
_output_shapes	
:�
�
Mtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_2_grad/Mul_1MulItraining/Adam/gradients/batch_normalization_3/cond/batchnorm/sub_grad/Neg5batch_normalization_3/cond/batchnorm/ReadVariableOp_1*
T0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/mul_2*
_output_shapes	
:�
�
Htraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/ShapeShapedense_3/Relu*
T0*
out_type0*8
_class.
,*loc:@batch_normalization_3/batchnorm/mul_1*
_output_shapes
:
�
Jtraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/Shape_1Shape#batch_normalization_3/batchnorm/mul*
T0*
out_type0*8
_class.
,*loc:@batch_normalization_3/batchnorm/mul_1*
_output_shapes
:
�
Xtraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsHtraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/ShapeJtraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/mul_1
�
Ftraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/MulMulJtraining/Adam/gradients/batch_normalization_3/batchnorm/add_1_grad/Reshape#batch_normalization_3/batchnorm/mul*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/mul_1*(
_output_shapes
:����������
�
Ftraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/SumSumFtraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/MulXtraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/mul_1*
_output_shapes
:
�
Jtraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/ReshapeReshapeFtraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/SumHtraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_3/batchnorm/mul_1*(
_output_shapes
:����������
�
Htraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/Mul_1Muldense_3/ReluJtraining/Adam/gradients/batch_normalization_3/batchnorm/add_1_grad/Reshape*(
_output_shapes
:����������*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/mul_1
�
Htraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/Sum_1SumHtraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/Mul_1Ztraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/mul_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
Ltraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/Reshape_1ReshapeHtraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/Sum_1Jtraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_3/batchnorm/mul_1*
_output_shapes	
:�
�
Dtraining/Adam/gradients/batch_normalization_3/batchnorm/sub_grad/NegNegLtraining/Adam/gradients/batch_normalization_3/batchnorm/add_1_grad/Reshape_1*
T0*6
_class,
*(loc:@batch_normalization_3/batchnorm/sub*
_output_shapes	
:�
�
 training/Adam/gradients/Switch_8Switchbatch_normalization_3/beta"batch_normalization_3/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_3/beta*
_output_shapes
: : 
�
"training/Adam/gradients/Identity_8Identity"training/Adam/gradients/Switch_8:1*
T0*-
_class#
!loc:@batch_normalization_3/beta*
_output_shapes
: 
�
'training/Adam/gradients/VariableShape_2VariableShape"training/Adam/gradients/Switch_8:1#^training/Adam/gradients/Identity_8*
out_type0*-
_class#
!loc:@batch_normalization_3/beta*
_output_shapes
:
�
%training/Adam/gradients/zeros_8/ConstConst#^training/Adam/gradients/Identity_8*
valueB
 *    *-
_class#
!loc:@batch_normalization_3/beta*
dtype0*
_output_shapes
: 
�
training/Adam/gradients/zeros_8Fill'training/Adam/gradients/VariableShape_2%training/Adam/gradients/zeros_8/Const*
T0*

index_type0*-
_class#
!loc:@batch_normalization_3/beta*#
_output_shapes
:���������
�
ctraining/Adam/gradients/batch_normalization_3/cond/batchnorm/ReadVariableOp_2/Switch_grad/cond_gradMergeQtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/Reshape_1training/Adam/gradients/zeros_8*
T0*-
_class#
!loc:@batch_normalization_3/beta*
N*%
_output_shapes
:���������: 
�
training/Adam/gradients/AddN_6AddNQtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/Reshape_1Mtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_2_grad/Mul_1*
T0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/mul_1*
N*
_output_shapes	
:�
�
Itraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_grad/MulMultraining/Adam/gradients/AddN_67batch_normalization_3/cond/batchnorm/mul/ReadVariableOp*
T0*;
_class1
/-loc:@batch_normalization_3/cond/batchnorm/mul*
_output_shapes	
:�
�
Ktraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_grad/Mul_1Multraining/Adam/gradients/AddN_6*batch_normalization_3/cond/batchnorm/Rsqrt*
_output_shapes	
:�*
T0*;
_class1
/-loc:@batch_normalization_3/cond/batchnorm/mul
�
Ftraining/Adam/gradients/batch_normalization_3/batchnorm/mul_2_grad/MulMulDtraining/Adam/gradients/batch_normalization_3/batchnorm/sub_grad/Neg#batch_normalization_3/batchnorm/mul*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/mul_2*
_output_shapes	
:�
�
Htraining/Adam/gradients/batch_normalization_3/batchnorm/mul_2_grad/Mul_1MulDtraining/Adam/gradients/batch_normalization_3/batchnorm/sub_grad/Neg%batch_normalization_3/moments/Squeeze*
_output_shapes	
:�*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/mul_2
�
training/Adam/gradients/AddN_7AddNctraining/Adam/gradients/batch_normalization_3/cond/batchnorm/ReadVariableOp_2/Switch_grad/cond_gradLtraining/Adam/gradients/batch_normalization_3/batchnorm/add_1_grad/Reshape_1*
N*
_output_shapes	
:�*
T0*-
_class#
!loc:@batch_normalization_3/beta
�
Htraining/Adam/gradients/batch_normalization_3/moments/Squeeze_grad/ShapeConst*
valueB"   �   *8
_class.
,*loc:@batch_normalization_3/moments/Squeeze*
dtype0*
_output_shapes
:
�
Jtraining/Adam/gradients/batch_normalization_3/moments/Squeeze_grad/ReshapeReshapeFtraining/Adam/gradients/batch_normalization_3/batchnorm/mul_2_grad/MulHtraining/Adam/gradients/batch_normalization_3/moments/Squeeze_grad/Shape*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_3/moments/Squeeze*
_output_shapes
:	�
�
training/Adam/gradients/AddN_8AddNLtraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/Reshape_1Htraining/Adam/gradients/batch_normalization_3/batchnorm/mul_2_grad/Mul_1*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/mul_1*
N*
_output_shapes	
:�
�
Dtraining/Adam/gradients/batch_normalization_3/batchnorm/mul_grad/MulMultraining/Adam/gradients/AddN_82batch_normalization_3/batchnorm/mul/ReadVariableOp*
T0*6
_class,
*(loc:@batch_normalization_3/batchnorm/mul*
_output_shapes	
:�
�
Ftraining/Adam/gradients/batch_normalization_3/batchnorm/mul_grad/Mul_1Multraining/Adam/gradients/AddN_8%batch_normalization_3/batchnorm/Rsqrt*
_output_shapes	
:�*
T0*6
_class,
*(loc:@batch_normalization_3/batchnorm/mul
�
 training/Adam/gradients/Switch_9Switchbatch_normalization_3/gamma"batch_normalization_3/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_3/gamma*
_output_shapes
: : 
�
"training/Adam/gradients/Identity_9Identity"training/Adam/gradients/Switch_9:1*
T0*.
_class$
" loc:@batch_normalization_3/gamma*
_output_shapes
: 
�
'training/Adam/gradients/VariableShape_3VariableShape"training/Adam/gradients/Switch_9:1#^training/Adam/gradients/Identity_9*
out_type0*.
_class$
" loc:@batch_normalization_3/gamma*
_output_shapes
:
�
%training/Adam/gradients/zeros_9/ConstConst#^training/Adam/gradients/Identity_9*
valueB
 *    *.
_class$
" loc:@batch_normalization_3/gamma*
dtype0*
_output_shapes
: 
�
training/Adam/gradients/zeros_9Fill'training/Adam/gradients/VariableShape_3%training/Adam/gradients/zeros_9/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization_3/gamma*#
_output_shapes
:���������
�
etraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul/ReadVariableOp/Switch_grad/cond_gradMergeKtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_grad/Mul_1training/Adam/gradients/zeros_9*
N*%
_output_shapes
:���������: *
T0*.
_class$
" loc:@batch_normalization_3/gamma
�
Ltraining/Adam/gradients/batch_normalization_3/batchnorm/Rsqrt_grad/RsqrtGrad	RsqrtGrad%batch_normalization_3/batchnorm/RsqrtDtraining/Adam/gradients/batch_normalization_3/batchnorm/mul_grad/Mul*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/Rsqrt*
_output_shapes	
:�
�
Vtraining/Adam/gradients/batch_normalization_3/batchnorm/add_grad/Sum/reduction_indicesConst*
valueB: *6
_class,
*(loc:@batch_normalization_3/batchnorm/add*
dtype0*
_output_shapes
:
�
Dtraining/Adam/gradients/batch_normalization_3/batchnorm/add_grad/SumSumLtraining/Adam/gradients/batch_normalization_3/batchnorm/Rsqrt_grad/RsqrtGradVtraining/Adam/gradients/batch_normalization_3/batchnorm/add_grad/Sum/reduction_indices*
T0*6
_class,
*(loc:@batch_normalization_3/batchnorm/add*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
Ntraining/Adam/gradients/batch_normalization_3/batchnorm/add_grad/Reshape/shapeConst*
valueB *6
_class,
*(loc:@batch_normalization_3/batchnorm/add*
dtype0*
_output_shapes
: 
�
Htraining/Adam/gradients/batch_normalization_3/batchnorm/add_grad/ReshapeReshapeDtraining/Adam/gradients/batch_normalization_3/batchnorm/add_grad/SumNtraining/Adam/gradients/batch_normalization_3/batchnorm/add_grad/Reshape/shape*
T0*
Tshape0*6
_class,
*(loc:@batch_normalization_3/batchnorm/add*
_output_shapes
: 
�
training/Adam/gradients/AddN_9AddNetraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul/ReadVariableOp/Switch_grad/cond_gradFtraining/Adam/gradients/batch_normalization_3/batchnorm/mul_grad/Mul_1*
T0*.
_class$
" loc:@batch_normalization_3/gamma*
N*
_output_shapes	
:�
�
Jtraining/Adam/gradients/batch_normalization_3/moments/Squeeze_1_grad/ShapeConst*
valueB"   �   *:
_class0
.,loc:@batch_normalization_3/moments/Squeeze_1*
dtype0*
_output_shapes
:
�
Ltraining/Adam/gradients/batch_normalization_3/moments/Squeeze_1_grad/ReshapeReshapeLtraining/Adam/gradients/batch_normalization_3/batchnorm/Rsqrt_grad/RsqrtGradJtraining/Adam/gradients/batch_normalization_3/moments/Squeeze_1_grad/Shape*
T0*
Tshape0*:
_class0
.,loc:@batch_normalization_3/moments/Squeeze_1*
_output_shapes
:	�
�
Itraining/Adam/gradients/batch_normalization_3/moments/variance_grad/ShapeShape/batch_normalization_3/moments/SquaredDifference*
T0*
out_type0*9
_class/
-+loc:@batch_normalization_3/moments/variance*
_output_shapes
:
�
Htraining/Adam/gradients/batch_normalization_3/moments/variance_grad/SizeConst*
value	B :*9
_class/
-+loc:@batch_normalization_3/moments/variance*
dtype0*
_output_shapes
: 
�
Gtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/addAddV28batch_normalization_3/moments/variance/reduction_indicesHtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Size*
T0*9
_class/
-+loc:@batch_normalization_3/moments/variance*
_output_shapes
:
�
Gtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/modFloorModGtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/addHtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Size*
_output_shapes
:*
T0*9
_class/
-+loc:@batch_normalization_3/moments/variance
�
Ktraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Shape_1Const*
valueB:*9
_class/
-+loc:@batch_normalization_3/moments/variance*
dtype0*
_output_shapes
:
�
Otraining/Adam/gradients/batch_normalization_3/moments/variance_grad/range/startConst*
value	B : *9
_class/
-+loc:@batch_normalization_3/moments/variance*
dtype0*
_output_shapes
: 
�
Otraining/Adam/gradients/batch_normalization_3/moments/variance_grad/range/deltaConst*
value	B :*9
_class/
-+loc:@batch_normalization_3/moments/variance*
dtype0*
_output_shapes
: 
�
Itraining/Adam/gradients/batch_normalization_3/moments/variance_grad/rangeRangeOtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/range/startHtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/SizeOtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/range/delta*

Tidx0*9
_class/
-+loc:@batch_normalization_3/moments/variance*
_output_shapes
:
�
Ntraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Fill/valueConst*
value	B :*9
_class/
-+loc:@batch_normalization_3/moments/variance*
dtype0*
_output_shapes
: 
�
Htraining/Adam/gradients/batch_normalization_3/moments/variance_grad/FillFillKtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Shape_1Ntraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Fill/value*
T0*

index_type0*9
_class/
-+loc:@batch_normalization_3/moments/variance*
_output_shapes
:
�
Qtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/DynamicStitchDynamicStitchItraining/Adam/gradients/batch_normalization_3/moments/variance_grad/rangeGtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/modItraining/Adam/gradients/batch_normalization_3/moments/variance_grad/ShapeHtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Fill*
T0*9
_class/
-+loc:@batch_normalization_3/moments/variance*
N*
_output_shapes
:
�
Mtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Maximum/yConst*
value	B :*9
_class/
-+loc:@batch_normalization_3/moments/variance*
dtype0*
_output_shapes
: 
�
Ktraining/Adam/gradients/batch_normalization_3/moments/variance_grad/MaximumMaximumQtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/DynamicStitchMtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Maximum/y*
T0*9
_class/
-+loc:@batch_normalization_3/moments/variance*
_output_shapes
:
�
Ltraining/Adam/gradients/batch_normalization_3/moments/variance_grad/floordivFloorDivItraining/Adam/gradients/batch_normalization_3/moments/variance_grad/ShapeKtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Maximum*
T0*9
_class/
-+loc:@batch_normalization_3/moments/variance*
_output_shapes
:
�
Ktraining/Adam/gradients/batch_normalization_3/moments/variance_grad/ReshapeReshapeLtraining/Adam/gradients/batch_normalization_3/moments/Squeeze_1_grad/ReshapeQtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/DynamicStitch*
T0*
Tshape0*9
_class/
-+loc:@batch_normalization_3/moments/variance*0
_output_shapes
:������������������
�
Htraining/Adam/gradients/batch_normalization_3/moments/variance_grad/TileTileKtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/ReshapeLtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/floordiv*

Tmultiples0*
T0*9
_class/
-+loc:@batch_normalization_3/moments/variance*0
_output_shapes
:������������������
�
Ktraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Shape_2Shape/batch_normalization_3/moments/SquaredDifference*
T0*
out_type0*9
_class/
-+loc:@batch_normalization_3/moments/variance*
_output_shapes
:
�
Ktraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Shape_3Const*
valueB"   �   *9
_class/
-+loc:@batch_normalization_3/moments/variance*
dtype0*
_output_shapes
:
�
Itraining/Adam/gradients/batch_normalization_3/moments/variance_grad/ConstConst*
valueB: *9
_class/
-+loc:@batch_normalization_3/moments/variance*
dtype0*
_output_shapes
:
�
Htraining/Adam/gradients/batch_normalization_3/moments/variance_grad/ProdProdKtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Shape_2Itraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Const*
T0*9
_class/
-+loc:@batch_normalization_3/moments/variance*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
Ktraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Const_1Const*
valueB: *9
_class/
-+loc:@batch_normalization_3/moments/variance*
dtype0*
_output_shapes
:
�
Jtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Prod_1ProdKtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Shape_3Ktraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Const_1*

Tidx0*
	keep_dims( *
T0*9
_class/
-+loc:@batch_normalization_3/moments/variance*
_output_shapes
: 
�
Otraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Maximum_1/yConst*
value	B :*9
_class/
-+loc:@batch_normalization_3/moments/variance*
dtype0*
_output_shapes
: 
�
Mtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Maximum_1MaximumJtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Prod_1Otraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Maximum_1/y*
T0*9
_class/
-+loc:@batch_normalization_3/moments/variance*
_output_shapes
: 
�
Ntraining/Adam/gradients/batch_normalization_3/moments/variance_grad/floordiv_1FloorDivHtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/ProdMtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Maximum_1*
T0*9
_class/
-+loc:@batch_normalization_3/moments/variance*
_output_shapes
: 
�
Htraining/Adam/gradients/batch_normalization_3/moments/variance_grad/CastCastNtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/floordiv_1*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0*9
_class/
-+loc:@batch_normalization_3/moments/variance
�
Ktraining/Adam/gradients/batch_normalization_3/moments/variance_grad/truedivRealDivHtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/TileHtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Cast*(
_output_shapes
:����������*
T0*9
_class/
-+loc:@batch_normalization_3/moments/variance
�
Straining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/scalarConstL^training/Adam/gradients/batch_normalization_3/moments/variance_grad/truediv*
valueB
 *   @*B
_class8
64loc:@batch_normalization_3/moments/SquaredDifference*
dtype0*
_output_shapes
: 
�
Ptraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/MulMulStraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/scalarKtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/truediv*
T0*B
_class8
64loc:@batch_normalization_3/moments/SquaredDifference*(
_output_shapes
:����������
�
Ptraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/subSubdense_3/Relu*batch_normalization_3/moments/StopGradientL^training/Adam/gradients/batch_normalization_3/moments/variance_grad/truediv*
T0*B
_class8
64loc:@batch_normalization_3/moments/SquaredDifference*(
_output_shapes
:����������
�
Rtraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/mul_1MulPtraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/MulPtraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/sub*
T0*B
_class8
64loc:@batch_normalization_3/moments/SquaredDifference*(
_output_shapes
:����������
�
Rtraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/ShapeShapedense_3/Relu*
T0*
out_type0*B
_class8
64loc:@batch_normalization_3/moments/SquaredDifference*
_output_shapes
:
�
Ttraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/Shape_1Shape*batch_normalization_3/moments/StopGradient*
T0*
out_type0*B
_class8
64loc:@batch_normalization_3/moments/SquaredDifference*
_output_shapes
:
�
btraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgsRtraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/ShapeTtraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/Shape_1*
T0*B
_class8
64loc:@batch_normalization_3/moments/SquaredDifference*2
_output_shapes 
:���������:���������
�
Ptraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/SumSumRtraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/mul_1btraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/BroadcastGradientArgs*
T0*B
_class8
64loc:@batch_normalization_3/moments/SquaredDifference*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
Ttraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/ReshapeReshapePtraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/SumRtraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/Shape*
T0*
Tshape0*B
_class8
64loc:@batch_normalization_3/moments/SquaredDifference*(
_output_shapes
:����������
�
Rtraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/Sum_1SumRtraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/mul_1dtraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*B
_class8
64loc:@batch_normalization_3/moments/SquaredDifference*
_output_shapes
:
�
Vtraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/Reshape_1ReshapeRtraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/Sum_1Ttraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/Shape_1*
T0*
Tshape0*B
_class8
64loc:@batch_normalization_3/moments/SquaredDifference*
_output_shapes
:	�
�
Ptraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/NegNegVtraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/Reshape_1*
T0*B
_class8
64loc:@batch_normalization_3/moments/SquaredDifference*
_output_shapes
:	�
�
Etraining/Adam/gradients/batch_normalization_3/moments/mean_grad/ShapeShapedense_3/Relu*
T0*
out_type0*5
_class+
)'loc:@batch_normalization_3/moments/mean*
_output_shapes
:
�
Dtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/SizeConst*
value	B :*5
_class+
)'loc:@batch_normalization_3/moments/mean*
dtype0*
_output_shapes
: 
�
Ctraining/Adam/gradients/batch_normalization_3/moments/mean_grad/addAddV24batch_normalization_3/moments/mean/reduction_indicesDtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/Size*
T0*5
_class+
)'loc:@batch_normalization_3/moments/mean*
_output_shapes
:
�
Ctraining/Adam/gradients/batch_normalization_3/moments/mean_grad/modFloorModCtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/addDtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/Size*
_output_shapes
:*
T0*5
_class+
)'loc:@batch_normalization_3/moments/mean
�
Gtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:*5
_class+
)'loc:@batch_normalization_3/moments/mean
�
Ktraining/Adam/gradients/batch_normalization_3/moments/mean_grad/range/startConst*
value	B : *5
_class+
)'loc:@batch_normalization_3/moments/mean*
dtype0*
_output_shapes
: 
�
Ktraining/Adam/gradients/batch_normalization_3/moments/mean_grad/range/deltaConst*
value	B :*5
_class+
)'loc:@batch_normalization_3/moments/mean*
dtype0*
_output_shapes
: 
�
Etraining/Adam/gradients/batch_normalization_3/moments/mean_grad/rangeRangeKtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/range/startDtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/SizeKtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/range/delta*5
_class+
)'loc:@batch_normalization_3/moments/mean*
_output_shapes
:*

Tidx0
�
Jtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/Fill/valueConst*
value	B :*5
_class+
)'loc:@batch_normalization_3/moments/mean*
dtype0*
_output_shapes
: 
�
Dtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/FillFillGtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/Shape_1Jtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/Fill/value*
T0*

index_type0*5
_class+
)'loc:@batch_normalization_3/moments/mean*
_output_shapes
:
�
Mtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/DynamicStitchDynamicStitchEtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/rangeCtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/modEtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/ShapeDtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/Fill*
T0*5
_class+
)'loc:@batch_normalization_3/moments/mean*
N*
_output_shapes
:
�
Itraining/Adam/gradients/batch_normalization_3/moments/mean_grad/Maximum/yConst*
value	B :*5
_class+
)'loc:@batch_normalization_3/moments/mean*
dtype0*
_output_shapes
: 
�
Gtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/MaximumMaximumMtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/DynamicStitchItraining/Adam/gradients/batch_normalization_3/moments/mean_grad/Maximum/y*
T0*5
_class+
)'loc:@batch_normalization_3/moments/mean*
_output_shapes
:
�
Htraining/Adam/gradients/batch_normalization_3/moments/mean_grad/floordivFloorDivEtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/ShapeGtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/Maximum*
T0*5
_class+
)'loc:@batch_normalization_3/moments/mean*
_output_shapes
:
�
Gtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/ReshapeReshapeJtraining/Adam/gradients/batch_normalization_3/moments/Squeeze_grad/ReshapeMtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/DynamicStitch*
T0*
Tshape0*5
_class+
)'loc:@batch_normalization_3/moments/mean*0
_output_shapes
:������������������
�
Dtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/TileTileGtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/ReshapeHtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/floordiv*

Tmultiples0*
T0*5
_class+
)'loc:@batch_normalization_3/moments/mean*0
_output_shapes
:������������������
�
Gtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/Shape_2Shapedense_3/Relu*
_output_shapes
:*
T0*
out_type0*5
_class+
)'loc:@batch_normalization_3/moments/mean
�
Gtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/Shape_3Const*
valueB"   �   *5
_class+
)'loc:@batch_normalization_3/moments/mean*
dtype0*
_output_shapes
:
�
Etraining/Adam/gradients/batch_normalization_3/moments/mean_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: *5
_class+
)'loc:@batch_normalization_3/moments/mean
�
Dtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/ProdProdGtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/Shape_2Etraining/Adam/gradients/batch_normalization_3/moments/mean_grad/Const*
T0*5
_class+
)'loc:@batch_normalization_3/moments/mean*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
Gtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/Const_1Const*
valueB: *5
_class+
)'loc:@batch_normalization_3/moments/mean*
dtype0*
_output_shapes
:
�
Ftraining/Adam/gradients/batch_normalization_3/moments/mean_grad/Prod_1ProdGtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/Shape_3Gtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/Const_1*
T0*5
_class+
)'loc:@batch_normalization_3/moments/mean*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
Ktraining/Adam/gradients/batch_normalization_3/moments/mean_grad/Maximum_1/yConst*
dtype0*
_output_shapes
: *
value	B :*5
_class+
)'loc:@batch_normalization_3/moments/mean
�
Itraining/Adam/gradients/batch_normalization_3/moments/mean_grad/Maximum_1MaximumFtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/Prod_1Ktraining/Adam/gradients/batch_normalization_3/moments/mean_grad/Maximum_1/y*
T0*5
_class+
)'loc:@batch_normalization_3/moments/mean*
_output_shapes
: 
�
Jtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/floordiv_1FloorDivDtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/ProdItraining/Adam/gradients/batch_normalization_3/moments/mean_grad/Maximum_1*
T0*5
_class+
)'loc:@batch_normalization_3/moments/mean*
_output_shapes
: 
�
Dtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/CastCastJtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/floordiv_1*

SrcT0*5
_class+
)'loc:@batch_normalization_3/moments/mean*
Truncate( *
_output_shapes
: *

DstT0
�
Gtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/truedivRealDivDtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/TileDtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/Cast*
T0*5
_class+
)'loc:@batch_normalization_3/moments/mean*(
_output_shapes
:����������
�
training/Adam/gradients/AddN_10AddNXtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_1/Switch_grad/cond_gradJtraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/ReshapeTtraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/ReshapeGtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/truediv*
T0*
_class
loc:@dense_3/Relu*
N*(
_output_shapes
:����������
�
2training/Adam/gradients/dense_3/Relu_grad/ReluGradReluGradtraining/Adam/gradients/AddN_10dense_3/Relu*
T0*
_class
loc:@dense_3/Relu*(
_output_shapes
:����������
�
8training/Adam/gradients/dense_3/BiasAdd_grad/BiasAddGradBiasAddGrad2training/Adam/gradients/dense_3/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:�*
T0*"
_class
loc:@dense_3/BiasAdd
�
2training/Adam/gradients/dense_3/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_3/Relu_grad/ReluGraddense_3/MatMul/ReadVariableOp*
transpose_b(*
T0*!
_class
loc:@dense_3/MatMul*(
_output_shapes
:����������*
transpose_a( 
�
4training/Adam/gradients/dense_3/MatMul_grad/MatMul_1MatMuldropout_1/cond/Merge2training/Adam/gradients/dense_3/Relu_grad/ReluGrad*
T0*!
_class
loc:@dense_3/MatMul* 
_output_shapes
:
��*
transpose_a(*
transpose_b( 
�
;training/Adam/gradients/dropout_1/cond/Merge_grad/cond_gradSwitch2training/Adam/gradients/dense_3/MatMul_grad/MatMuldropout_1/cond/pred_id*
T0*!
_class
loc:@dense_3/MatMul*<
_output_shapes*
(:����������:����������
�
!training/Adam/gradients/Switch_10Switch batch_normalization_2/cond/Mergedropout_1/cond/pred_id*
T0*3
_class)
'%loc:@batch_normalization_2/cond/Merge*<
_output_shapes*
(:����������:����������
�
#training/Adam/gradients/Identity_10Identity#training/Adam/gradients/Switch_10:1*
T0*3
_class)
'%loc:@batch_normalization_2/cond/Merge*(
_output_shapes
:����������
�
training/Adam/gradients/Shape_7Shape#training/Adam/gradients/Switch_10:1*
_output_shapes
:*
T0*
out_type0*3
_class)
'%loc:@batch_normalization_2/cond/Merge
�
&training/Adam/gradients/zeros_10/ConstConst$^training/Adam/gradients/Identity_10*
valueB
 *    *3
_class)
'%loc:@batch_normalization_2/cond/Merge*
dtype0*
_output_shapes
: 
�
 training/Adam/gradients/zeros_10Filltraining/Adam/gradients/Shape_7&training/Adam/gradients/zeros_10/Const*(
_output_shapes
:����������*
T0*

index_type0*3
_class)
'%loc:@batch_normalization_2/cond/Merge
�
>training/Adam/gradients/dropout_1/cond/Switch_1_grad/cond_gradMerge;training/Adam/gradients/dropout_1/cond/Merge_grad/cond_grad training/Adam/gradients/zeros_10*
N**
_output_shapes
:����������: *
T0*3
_class)
'%loc:@batch_normalization_2/cond/Merge
�
?training/Adam/gradients/dropout_1/cond/dropout/mul_1_grad/ShapeShapedropout_1/cond/dropout/mul*
_output_shapes
:*
T0*
out_type0*/
_class%
#!loc:@dropout_1/cond/dropout/mul_1
�
Atraining/Adam/gradients/dropout_1/cond/dropout/mul_1_grad/Shape_1Shapedropout_1/cond/dropout/Cast*
_output_shapes
:*
T0*
out_type0*/
_class%
#!loc:@dropout_1/cond/dropout/mul_1
�
Otraining/Adam/gradients/dropout_1/cond/dropout/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs?training/Adam/gradients/dropout_1/cond/dropout/mul_1_grad/ShapeAtraining/Adam/gradients/dropout_1/cond/dropout/mul_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0*/
_class%
#!loc:@dropout_1/cond/dropout/mul_1
�
=training/Adam/gradients/dropout_1/cond/dropout/mul_1_grad/MulMul=training/Adam/gradients/dropout_1/cond/Merge_grad/cond_grad:1dropout_1/cond/dropout/Cast*
T0*/
_class%
#!loc:@dropout_1/cond/dropout/mul_1*(
_output_shapes
:����������
�
=training/Adam/gradients/dropout_1/cond/dropout/mul_1_grad/SumSum=training/Adam/gradients/dropout_1/cond/dropout/mul_1_grad/MulOtraining/Adam/gradients/dropout_1/cond/dropout/mul_1_grad/BroadcastGradientArgs*
T0*/
_class%
#!loc:@dropout_1/cond/dropout/mul_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
Atraining/Adam/gradients/dropout_1/cond/dropout/mul_1_grad/ReshapeReshape=training/Adam/gradients/dropout_1/cond/dropout/mul_1_grad/Sum?training/Adam/gradients/dropout_1/cond/dropout/mul_1_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0*/
_class%
#!loc:@dropout_1/cond/dropout/mul_1
�
?training/Adam/gradients/dropout_1/cond/dropout/mul_1_grad/Mul_1Muldropout_1/cond/dropout/mul=training/Adam/gradients/dropout_1/cond/Merge_grad/cond_grad:1*
T0*/
_class%
#!loc:@dropout_1/cond/dropout/mul_1*(
_output_shapes
:����������
�
?training/Adam/gradients/dropout_1/cond/dropout/mul_1_grad/Sum_1Sum?training/Adam/gradients/dropout_1/cond/dropout/mul_1_grad/Mul_1Qtraining/Adam/gradients/dropout_1/cond/dropout/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*/
_class%
#!loc:@dropout_1/cond/dropout/mul_1
�
Ctraining/Adam/gradients/dropout_1/cond/dropout/mul_1_grad/Reshape_1Reshape?training/Adam/gradients/dropout_1/cond/dropout/mul_1_grad/Sum_1Atraining/Adam/gradients/dropout_1/cond/dropout/mul_1_grad/Shape_1*
T0*
Tshape0*/
_class%
#!loc:@dropout_1/cond/dropout/mul_1*(
_output_shapes
:����������
�
=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/ShapeShape%dropout_1/cond/dropout/Shape/Switch:1*
_output_shapes
:*
T0*
out_type0*-
_class#
!loc:@dropout_1/cond/dropout/mul
�
?training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Shape_1Shapedropout_1/cond/dropout/truediv*
T0*
out_type0*-
_class#
!loc:@dropout_1/cond/dropout/mul*
_output_shapes
: 
�
Mtraining/Adam/gradients/dropout_1/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Shape?training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Shape_1*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*2
_output_shapes 
:���������:���������
�
;training/Adam/gradients/dropout_1/cond/dropout/mul_grad/MulMulAtraining/Adam/gradients/dropout_1/cond/dropout/mul_1_grad/Reshapedropout_1/cond/dropout/truediv*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*(
_output_shapes
:����������
�
;training/Adam/gradients/dropout_1/cond/dropout/mul_grad/SumSum;training/Adam/gradients/dropout_1/cond/dropout/mul_grad/MulMtraining/Adam/gradients/dropout_1/cond/dropout/mul_grad/BroadcastGradientArgs*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
?training/Adam/gradients/dropout_1/cond/dropout/mul_grad/ReshapeReshape;training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Sum=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Shape*
T0*
Tshape0*-
_class#
!loc:@dropout_1/cond/dropout/mul*(
_output_shapes
:����������
�
=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Mul_1Mul%dropout_1/cond/dropout/Shape/Switch:1Atraining/Adam/gradients/dropout_1/cond/dropout/mul_1_grad/Reshape*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*(
_output_shapes
:����������
�
=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Sum_1Sum=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Mul_1Otraining/Adam/gradients/dropout_1/cond/dropout/mul_grad/BroadcastGradientArgs:1*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
Atraining/Adam/gradients/dropout_1/cond/dropout/mul_grad/Reshape_1Reshape=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Sum_1?training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Shape_1*
T0*
Tshape0*-
_class#
!loc:@dropout_1/cond/dropout/mul*
_output_shapes
: 
�
!training/Adam/gradients/Switch_11Switch batch_normalization_2/cond/Mergedropout_1/cond/pred_id*
T0*3
_class)
'%loc:@batch_normalization_2/cond/Merge*<
_output_shapes*
(:����������:����������
�
#training/Adam/gradients/Identity_11Identity!training/Adam/gradients/Switch_11*
T0*3
_class)
'%loc:@batch_normalization_2/cond/Merge*(
_output_shapes
:����������
�
training/Adam/gradients/Shape_8Shape!training/Adam/gradients/Switch_11*
T0*
out_type0*3
_class)
'%loc:@batch_normalization_2/cond/Merge*
_output_shapes
:
�
&training/Adam/gradients/zeros_11/ConstConst$^training/Adam/gradients/Identity_11*
valueB
 *    *3
_class)
'%loc:@batch_normalization_2/cond/Merge*
dtype0*
_output_shapes
: 
�
 training/Adam/gradients/zeros_11Filltraining/Adam/gradients/Shape_8&training/Adam/gradients/zeros_11/Const*
T0*

index_type0*3
_class)
'%loc:@batch_normalization_2/cond/Merge*(
_output_shapes
:����������
�
Jtraining/Adam/gradients/dropout_1/cond/dropout/Shape/Switch_grad/cond_gradMerge training/Adam/gradients/zeros_11?training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Reshape*
T0*3
_class)
'%loc:@batch_normalization_2/cond/Merge*
N**
_output_shapes
:����������: 
�
training/Adam/gradients/AddN_11AddN>training/Adam/gradients/dropout_1/cond/Switch_1_grad/cond_gradJtraining/Adam/gradients/dropout_1/cond/dropout/Shape/Switch_grad/cond_grad*
T0*3
_class)
'%loc:@batch_normalization_2/cond/Merge*
N*(
_output_shapes
:����������
�
Gtraining/Adam/gradients/batch_normalization_2/cond/Merge_grad/cond_gradSwitchtraining/Adam/gradients/AddN_11"batch_normalization_2/cond/pred_id*
T0*3
_class)
'%loc:@batch_normalization_2/cond/Merge*<
_output_shapes*
(:����������:����������
�
Mtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/ShapeShape*batch_normalization_2/cond/batchnorm/mul_1*
T0*
out_type0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/add_1*
_output_shapes
:
�
Otraining/Adam/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/Shape_1Shape(batch_normalization_2/cond/batchnorm/sub*
T0*
out_type0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/add_1*
_output_shapes
:
�
]training/Adam/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsMtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/ShapeOtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/add_1
�
Ktraining/Adam/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/SumSumGtraining/Adam/gradients/batch_normalization_2/cond/Merge_grad/cond_grad]training/Adam/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/BroadcastGradientArgs*
T0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/add_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
Otraining/Adam/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/ReshapeReshapeKtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/SumMtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/add_1
�
Mtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/Sum_1SumGtraining/Adam/gradients/batch_normalization_2/cond/Merge_grad/cond_grad_training/Adam/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/add_1*
_output_shapes
:
�
Qtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/Reshape_1ReshapeMtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/Sum_1Otraining/Adam/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/Shape_1*
_output_shapes	
:�*
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/add_1
�
!training/Adam/gradients/Switch_12Switch%batch_normalization_2/batchnorm/add_1"batch_normalization_2/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/add_1*<
_output_shapes*
(:����������:����������
�
#training/Adam/gradients/Identity_12Identity!training/Adam/gradients/Switch_12*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/add_1*(
_output_shapes
:����������
�
training/Adam/gradients/Shape_9Shape!training/Adam/gradients/Switch_12*
T0*
out_type0*8
_class.
,*loc:@batch_normalization_2/batchnorm/add_1*
_output_shapes
:
�
&training/Adam/gradients/zeros_12/ConstConst$^training/Adam/gradients/Identity_12*
valueB
 *    *8
_class.
,*loc:@batch_normalization_2/batchnorm/add_1*
dtype0*
_output_shapes
: 
�
 training/Adam/gradients/zeros_12Filltraining/Adam/gradients/Shape_9&training/Adam/gradients/zeros_12/Const*
T0*

index_type0*8
_class.
,*loc:@batch_normalization_2/batchnorm/add_1*(
_output_shapes
:����������
�
Jtraining/Adam/gradients/batch_normalization_2/cond/Switch_1_grad/cond_gradMerge training/Adam/gradients/zeros_12Itraining/Adam/gradients/batch_normalization_2/cond/Merge_grad/cond_grad:1*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/add_1*
N**
_output_shapes
:����������: 
�
Mtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/ShapeShape1batch_normalization_2/cond/batchnorm/mul_1/Switch*
T0*
out_type0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/mul_1*
_output_shapes
:
�
Otraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/Shape_1Shape(batch_normalization_2/cond/batchnorm/mul*
_output_shapes
:*
T0*
out_type0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/mul_1
�
]training/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsMtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/ShapeOtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/Shape_1*
T0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/mul_1*2
_output_shapes 
:���������:���������
�
Ktraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/MulMulOtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/Reshape(batch_normalization_2/cond/batchnorm/mul*(
_output_shapes
:����������*
T0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/mul_1
�
Ktraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/SumSumKtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/Mul]training/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/mul_1
�
Otraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/ReshapeReshapeKtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/SumMtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/mul_1*(
_output_shapes
:����������
�
Mtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/Mul_1Mul1batch_normalization_2/cond/batchnorm/mul_1/SwitchOtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/Reshape*
T0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/mul_1*(
_output_shapes
:����������
�
Mtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/Sum_1SumMtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/Mul_1_training/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/mul_1
�
Qtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/Reshape_1ReshapeMtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/Sum_1Otraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/mul_1*
_output_shapes	
:�
�
Itraining/Adam/gradients/batch_normalization_2/cond/batchnorm/sub_grad/NegNegQtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:�*
T0*;
_class1
/-loc:@batch_normalization_2/cond/batchnorm/sub
�
Htraining/Adam/gradients/batch_normalization_2/batchnorm/add_1_grad/ShapeShape%batch_normalization_2/batchnorm/mul_1*
_output_shapes
:*
T0*
out_type0*8
_class.
,*loc:@batch_normalization_2/batchnorm/add_1
�
Jtraining/Adam/gradients/batch_normalization_2/batchnorm/add_1_grad/Shape_1Shape#batch_normalization_2/batchnorm/sub*
T0*
out_type0*8
_class.
,*loc:@batch_normalization_2/batchnorm/add_1*
_output_shapes
:
�
Xtraining/Adam/gradients/batch_normalization_2/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsHtraining/Adam/gradients/batch_normalization_2/batchnorm/add_1_grad/ShapeJtraining/Adam/gradients/batch_normalization_2/batchnorm/add_1_grad/Shape_1*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/add_1*2
_output_shapes 
:���������:���������
�
Ftraining/Adam/gradients/batch_normalization_2/batchnorm/add_1_grad/SumSumJtraining/Adam/gradients/batch_normalization_2/cond/Switch_1_grad/cond_gradXtraining/Adam/gradients/batch_normalization_2/batchnorm/add_1_grad/BroadcastGradientArgs*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/add_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
Jtraining/Adam/gradients/batch_normalization_2/batchnorm/add_1_grad/ReshapeReshapeFtraining/Adam/gradients/batch_normalization_2/batchnorm/add_1_grad/SumHtraining/Adam/gradients/batch_normalization_2/batchnorm/add_1_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_2/batchnorm/add_1
�
Htraining/Adam/gradients/batch_normalization_2/batchnorm/add_1_grad/Sum_1SumJtraining/Adam/gradients/batch_normalization_2/cond/Switch_1_grad/cond_gradZtraining/Adam/gradients/batch_normalization_2/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/add_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
Ltraining/Adam/gradients/batch_normalization_2/batchnorm/add_1_grad/Reshape_1ReshapeHtraining/Adam/gradients/batch_normalization_2/batchnorm/add_1_grad/Sum_1Jtraining/Adam/gradients/batch_normalization_2/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_2/batchnorm/add_1*
_output_shapes	
:�
�
!training/Adam/gradients/Switch_13Switchdense_2/Relu"batch_normalization_2/cond/pred_id*
T0*
_class
loc:@dense_2/Relu*<
_output_shapes*
(:����������:����������
�
#training/Adam/gradients/Identity_13Identity#training/Adam/gradients/Switch_13:1*
T0*
_class
loc:@dense_2/Relu*(
_output_shapes
:����������
�
 training/Adam/gradients/Shape_10Shape#training/Adam/gradients/Switch_13:1*
T0*
out_type0*
_class
loc:@dense_2/Relu*
_output_shapes
:
�
&training/Adam/gradients/zeros_13/ConstConst$^training/Adam/gradients/Identity_13*
valueB
 *    *
_class
loc:@dense_2/Relu*
dtype0*
_output_shapes
: 
�
 training/Adam/gradients/zeros_13Fill training/Adam/gradients/Shape_10&training/Adam/gradients/zeros_13/Const*
T0*

index_type0*
_class
loc:@dense_2/Relu*(
_output_shapes
:����������
�
Xtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1/Switch_grad/cond_gradMergeOtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/Reshape training/Adam/gradients/zeros_13*
T0*
_class
loc:@dense_2/Relu*
N**
_output_shapes
:����������: 
�
Ktraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_2_grad/MulMulItraining/Adam/gradients/batch_normalization_2/cond/batchnorm/sub_grad/Neg(batch_normalization_2/cond/batchnorm/mul*
T0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/mul_2*
_output_shapes	
:�
�
Mtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_2_grad/Mul_1MulItraining/Adam/gradients/batch_normalization_2/cond/batchnorm/sub_grad/Neg5batch_normalization_2/cond/batchnorm/ReadVariableOp_1*
T0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/mul_2*
_output_shapes	
:�
�
Htraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/ShapeShapedense_2/Relu*
T0*
out_type0*8
_class.
,*loc:@batch_normalization_2/batchnorm/mul_1*
_output_shapes
:
�
Jtraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/Shape_1Shape#batch_normalization_2/batchnorm/mul*
T0*
out_type0*8
_class.
,*loc:@batch_normalization_2/batchnorm/mul_1*
_output_shapes
:
�
Xtraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsHtraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/ShapeJtraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/Shape_1*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/mul_1*2
_output_shapes 
:���������:���������
�
Ftraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/MulMulJtraining/Adam/gradients/batch_normalization_2/batchnorm/add_1_grad/Reshape#batch_normalization_2/batchnorm/mul*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/mul_1*(
_output_shapes
:����������
�
Ftraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/SumSumFtraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/MulXtraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/mul_1*
_output_shapes
:
�
Jtraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/ReshapeReshapeFtraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/SumHtraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_2/batchnorm/mul_1*(
_output_shapes
:����������
�
Htraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/Mul_1Muldense_2/ReluJtraining/Adam/gradients/batch_normalization_2/batchnorm/add_1_grad/Reshape*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/mul_1*(
_output_shapes
:����������
�
Htraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/Sum_1SumHtraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/Mul_1Ztraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/mul_1
�
Ltraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/Reshape_1ReshapeHtraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/Sum_1Jtraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_2/batchnorm/mul_1*
_output_shapes	
:�
�
Dtraining/Adam/gradients/batch_normalization_2/batchnorm/sub_grad/NegNegLtraining/Adam/gradients/batch_normalization_2/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:�*
T0*6
_class,
*(loc:@batch_normalization_2/batchnorm/sub
�
!training/Adam/gradients/Switch_14Switchbatch_normalization_2/beta"batch_normalization_2/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_2/beta*
_output_shapes
: : 
�
#training/Adam/gradients/Identity_14Identity#training/Adam/gradients/Switch_14:1*
T0*-
_class#
!loc:@batch_normalization_2/beta*
_output_shapes
: 
�
'training/Adam/gradients/VariableShape_4VariableShape#training/Adam/gradients/Switch_14:1$^training/Adam/gradients/Identity_14*
out_type0*-
_class#
!loc:@batch_normalization_2/beta*
_output_shapes
:
�
&training/Adam/gradients/zeros_14/ConstConst$^training/Adam/gradients/Identity_14*
valueB
 *    *-
_class#
!loc:@batch_normalization_2/beta*
dtype0*
_output_shapes
: 
�
 training/Adam/gradients/zeros_14Fill'training/Adam/gradients/VariableShape_4&training/Adam/gradients/zeros_14/Const*
T0*

index_type0*-
_class#
!loc:@batch_normalization_2/beta*#
_output_shapes
:���������
�
ctraining/Adam/gradients/batch_normalization_2/cond/batchnorm/ReadVariableOp_2/Switch_grad/cond_gradMergeQtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/Reshape_1 training/Adam/gradients/zeros_14*
N*%
_output_shapes
:���������: *
T0*-
_class#
!loc:@batch_normalization_2/beta
�
training/Adam/gradients/AddN_12AddNQtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/Reshape_1Mtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_2_grad/Mul_1*
N*
_output_shapes	
:�*
T0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/mul_1
�
Itraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_grad/MulMultraining/Adam/gradients/AddN_127batch_normalization_2/cond/batchnorm/mul/ReadVariableOp*
T0*;
_class1
/-loc:@batch_normalization_2/cond/batchnorm/mul*
_output_shapes	
:�
�
Ktraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_grad/Mul_1Multraining/Adam/gradients/AddN_12*batch_normalization_2/cond/batchnorm/Rsqrt*
_output_shapes	
:�*
T0*;
_class1
/-loc:@batch_normalization_2/cond/batchnorm/mul
�
Ftraining/Adam/gradients/batch_normalization_2/batchnorm/mul_2_grad/MulMulDtraining/Adam/gradients/batch_normalization_2/batchnorm/sub_grad/Neg#batch_normalization_2/batchnorm/mul*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/mul_2*
_output_shapes	
:�
�
Htraining/Adam/gradients/batch_normalization_2/batchnorm/mul_2_grad/Mul_1MulDtraining/Adam/gradients/batch_normalization_2/batchnorm/sub_grad/Neg%batch_normalization_2/moments/Squeeze*
_output_shapes	
:�*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/mul_2
�
training/Adam/gradients/AddN_13AddNctraining/Adam/gradients/batch_normalization_2/cond/batchnorm/ReadVariableOp_2/Switch_grad/cond_gradLtraining/Adam/gradients/batch_normalization_2/batchnorm/add_1_grad/Reshape_1*
T0*-
_class#
!loc:@batch_normalization_2/beta*
N*
_output_shapes	
:�
�
Htraining/Adam/gradients/batch_normalization_2/moments/Squeeze_grad/ShapeConst*
valueB"   �   *8
_class.
,*loc:@batch_normalization_2/moments/Squeeze*
dtype0*
_output_shapes
:
�
Jtraining/Adam/gradients/batch_normalization_2/moments/Squeeze_grad/ReshapeReshapeFtraining/Adam/gradients/batch_normalization_2/batchnorm/mul_2_grad/MulHtraining/Adam/gradients/batch_normalization_2/moments/Squeeze_grad/Shape*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_2/moments/Squeeze*
_output_shapes
:	�
�
training/Adam/gradients/AddN_14AddNLtraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/Reshape_1Htraining/Adam/gradients/batch_normalization_2/batchnorm/mul_2_grad/Mul_1*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/mul_1*
N*
_output_shapes	
:�
�
Dtraining/Adam/gradients/batch_normalization_2/batchnorm/mul_grad/MulMultraining/Adam/gradients/AddN_142batch_normalization_2/batchnorm/mul/ReadVariableOp*
T0*6
_class,
*(loc:@batch_normalization_2/batchnorm/mul*
_output_shapes	
:�
�
Ftraining/Adam/gradients/batch_normalization_2/batchnorm/mul_grad/Mul_1Multraining/Adam/gradients/AddN_14%batch_normalization_2/batchnorm/Rsqrt*
T0*6
_class,
*(loc:@batch_normalization_2/batchnorm/mul*
_output_shapes	
:�
�
!training/Adam/gradients/Switch_15Switchbatch_normalization_2/gamma"batch_normalization_2/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_2/gamma*
_output_shapes
: : 
�
#training/Adam/gradients/Identity_15Identity#training/Adam/gradients/Switch_15:1*
T0*.
_class$
" loc:@batch_normalization_2/gamma*
_output_shapes
: 
�
'training/Adam/gradients/VariableShape_5VariableShape#training/Adam/gradients/Switch_15:1$^training/Adam/gradients/Identity_15*
out_type0*.
_class$
" loc:@batch_normalization_2/gamma*
_output_shapes
:
�
&training/Adam/gradients/zeros_15/ConstConst$^training/Adam/gradients/Identity_15*
valueB
 *    *.
_class$
" loc:@batch_normalization_2/gamma*
dtype0*
_output_shapes
: 
�
 training/Adam/gradients/zeros_15Fill'training/Adam/gradients/VariableShape_5&training/Adam/gradients/zeros_15/Const*#
_output_shapes
:���������*
T0*

index_type0*.
_class$
" loc:@batch_normalization_2/gamma
�
etraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul/ReadVariableOp/Switch_grad/cond_gradMergeKtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_grad/Mul_1 training/Adam/gradients/zeros_15*
T0*.
_class$
" loc:@batch_normalization_2/gamma*
N*%
_output_shapes
:���������: 
�
Ltraining/Adam/gradients/batch_normalization_2/batchnorm/Rsqrt_grad/RsqrtGrad	RsqrtGrad%batch_normalization_2/batchnorm/RsqrtDtraining/Adam/gradients/batch_normalization_2/batchnorm/mul_grad/Mul*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/Rsqrt*
_output_shapes	
:�
�
Vtraining/Adam/gradients/batch_normalization_2/batchnorm/add_grad/Sum/reduction_indicesConst*
valueB: *6
_class,
*(loc:@batch_normalization_2/batchnorm/add*
dtype0*
_output_shapes
:
�
Dtraining/Adam/gradients/batch_normalization_2/batchnorm/add_grad/SumSumLtraining/Adam/gradients/batch_normalization_2/batchnorm/Rsqrt_grad/RsqrtGradVtraining/Adam/gradients/batch_normalization_2/batchnorm/add_grad/Sum/reduction_indices*

Tidx0*
	keep_dims( *
T0*6
_class,
*(loc:@batch_normalization_2/batchnorm/add*
_output_shapes
: 
�
Ntraining/Adam/gradients/batch_normalization_2/batchnorm/add_grad/Reshape/shapeConst*
valueB *6
_class,
*(loc:@batch_normalization_2/batchnorm/add*
dtype0*
_output_shapes
: 
�
Htraining/Adam/gradients/batch_normalization_2/batchnorm/add_grad/ReshapeReshapeDtraining/Adam/gradients/batch_normalization_2/batchnorm/add_grad/SumNtraining/Adam/gradients/batch_normalization_2/batchnorm/add_grad/Reshape/shape*
T0*
Tshape0*6
_class,
*(loc:@batch_normalization_2/batchnorm/add*
_output_shapes
: 
�
training/Adam/gradients/AddN_15AddNetraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul/ReadVariableOp/Switch_grad/cond_gradFtraining/Adam/gradients/batch_normalization_2/batchnorm/mul_grad/Mul_1*
T0*.
_class$
" loc:@batch_normalization_2/gamma*
N*
_output_shapes	
:�
�
Jtraining/Adam/gradients/batch_normalization_2/moments/Squeeze_1_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"   �   *:
_class0
.,loc:@batch_normalization_2/moments/Squeeze_1
�
Ltraining/Adam/gradients/batch_normalization_2/moments/Squeeze_1_grad/ReshapeReshapeLtraining/Adam/gradients/batch_normalization_2/batchnorm/Rsqrt_grad/RsqrtGradJtraining/Adam/gradients/batch_normalization_2/moments/Squeeze_1_grad/Shape*
T0*
Tshape0*:
_class0
.,loc:@batch_normalization_2/moments/Squeeze_1*
_output_shapes
:	�
�
Itraining/Adam/gradients/batch_normalization_2/moments/variance_grad/ShapeShape/batch_normalization_2/moments/SquaredDifference*
T0*
out_type0*9
_class/
-+loc:@batch_normalization_2/moments/variance*
_output_shapes
:
�
Htraining/Adam/gradients/batch_normalization_2/moments/variance_grad/SizeConst*
value	B :*9
_class/
-+loc:@batch_normalization_2/moments/variance*
dtype0*
_output_shapes
: 
�
Gtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/addAddV28batch_normalization_2/moments/variance/reduction_indicesHtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/Size*
T0*9
_class/
-+loc:@batch_normalization_2/moments/variance*
_output_shapes
:
�
Gtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/modFloorModGtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/addHtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/Size*
_output_shapes
:*
T0*9
_class/
-+loc:@batch_normalization_2/moments/variance
�
Ktraining/Adam/gradients/batch_normalization_2/moments/variance_grad/Shape_1Const*
valueB:*9
_class/
-+loc:@batch_normalization_2/moments/variance*
dtype0*
_output_shapes
:
�
Otraining/Adam/gradients/batch_normalization_2/moments/variance_grad/range/startConst*
value	B : *9
_class/
-+loc:@batch_normalization_2/moments/variance*
dtype0*
_output_shapes
: 
�
Otraining/Adam/gradients/batch_normalization_2/moments/variance_grad/range/deltaConst*
value	B :*9
_class/
-+loc:@batch_normalization_2/moments/variance*
dtype0*
_output_shapes
: 
�
Itraining/Adam/gradients/batch_normalization_2/moments/variance_grad/rangeRangeOtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/range/startHtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/SizeOtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/range/delta*9
_class/
-+loc:@batch_normalization_2/moments/variance*
_output_shapes
:*

Tidx0
�
Ntraining/Adam/gradients/batch_normalization_2/moments/variance_grad/Fill/valueConst*
dtype0*
_output_shapes
: *
value	B :*9
_class/
-+loc:@batch_normalization_2/moments/variance
�
Htraining/Adam/gradients/batch_normalization_2/moments/variance_grad/FillFillKtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/Shape_1Ntraining/Adam/gradients/batch_normalization_2/moments/variance_grad/Fill/value*
T0*

index_type0*9
_class/
-+loc:@batch_normalization_2/moments/variance*
_output_shapes
:
�
Qtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/DynamicStitchDynamicStitchItraining/Adam/gradients/batch_normalization_2/moments/variance_grad/rangeGtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/modItraining/Adam/gradients/batch_normalization_2/moments/variance_grad/ShapeHtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/Fill*
T0*9
_class/
-+loc:@batch_normalization_2/moments/variance*
N*
_output_shapes
:
�
Mtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :*9
_class/
-+loc:@batch_normalization_2/moments/variance
�
Ktraining/Adam/gradients/batch_normalization_2/moments/variance_grad/MaximumMaximumQtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/DynamicStitchMtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/Maximum/y*
T0*9
_class/
-+loc:@batch_normalization_2/moments/variance*
_output_shapes
:
�
Ltraining/Adam/gradients/batch_normalization_2/moments/variance_grad/floordivFloorDivItraining/Adam/gradients/batch_normalization_2/moments/variance_grad/ShapeKtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/Maximum*
T0*9
_class/
-+loc:@batch_normalization_2/moments/variance*
_output_shapes
:
�
Ktraining/Adam/gradients/batch_normalization_2/moments/variance_grad/ReshapeReshapeLtraining/Adam/gradients/batch_normalization_2/moments/Squeeze_1_grad/ReshapeQtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/DynamicStitch*
T0*
Tshape0*9
_class/
-+loc:@batch_normalization_2/moments/variance*0
_output_shapes
:������������������
�
Htraining/Adam/gradients/batch_normalization_2/moments/variance_grad/TileTileKtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/ReshapeLtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/floordiv*
T0*9
_class/
-+loc:@batch_normalization_2/moments/variance*0
_output_shapes
:������������������*

Tmultiples0
�
Ktraining/Adam/gradients/batch_normalization_2/moments/variance_grad/Shape_2Shape/batch_normalization_2/moments/SquaredDifference*
T0*
out_type0*9
_class/
-+loc:@batch_normalization_2/moments/variance*
_output_shapes
:
�
Ktraining/Adam/gradients/batch_normalization_2/moments/variance_grad/Shape_3Const*
valueB"   �   *9
_class/
-+loc:@batch_normalization_2/moments/variance*
dtype0*
_output_shapes
:
�
Itraining/Adam/gradients/batch_normalization_2/moments/variance_grad/ConstConst*
valueB: *9
_class/
-+loc:@batch_normalization_2/moments/variance*
dtype0*
_output_shapes
:
�
Htraining/Adam/gradients/batch_normalization_2/moments/variance_grad/ProdProdKtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/Shape_2Itraining/Adam/gradients/batch_normalization_2/moments/variance_grad/Const*
T0*9
_class/
-+loc:@batch_normalization_2/moments/variance*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
Ktraining/Adam/gradients/batch_normalization_2/moments/variance_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: *9
_class/
-+loc:@batch_normalization_2/moments/variance
�
Jtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/Prod_1ProdKtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/Shape_3Ktraining/Adam/gradients/batch_normalization_2/moments/variance_grad/Const_1*
T0*9
_class/
-+loc:@batch_normalization_2/moments/variance*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
Otraining/Adam/gradients/batch_normalization_2/moments/variance_grad/Maximum_1/yConst*
value	B :*9
_class/
-+loc:@batch_normalization_2/moments/variance*
dtype0*
_output_shapes
: 
�
Mtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/Maximum_1MaximumJtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/Prod_1Otraining/Adam/gradients/batch_normalization_2/moments/variance_grad/Maximum_1/y*
_output_shapes
: *
T0*9
_class/
-+loc:@batch_normalization_2/moments/variance
�
Ntraining/Adam/gradients/batch_normalization_2/moments/variance_grad/floordiv_1FloorDivHtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/ProdMtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/Maximum_1*
T0*9
_class/
-+loc:@batch_normalization_2/moments/variance*
_output_shapes
: 
�
Htraining/Adam/gradients/batch_normalization_2/moments/variance_grad/CastCastNtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/floordiv_1*

SrcT0*9
_class/
-+loc:@batch_normalization_2/moments/variance*
Truncate( *
_output_shapes
: *

DstT0
�
Ktraining/Adam/gradients/batch_normalization_2/moments/variance_grad/truedivRealDivHtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/TileHtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/Cast*
T0*9
_class/
-+loc:@batch_normalization_2/moments/variance*(
_output_shapes
:����������
�
Straining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/scalarConstL^training/Adam/gradients/batch_normalization_2/moments/variance_grad/truediv*
valueB
 *   @*B
_class8
64loc:@batch_normalization_2/moments/SquaredDifference*
dtype0*
_output_shapes
: 
�
Ptraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/MulMulStraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/scalarKtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/truediv*
T0*B
_class8
64loc:@batch_normalization_2/moments/SquaredDifference*(
_output_shapes
:����������
�
Ptraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/subSubdense_2/Relu*batch_normalization_2/moments/StopGradientL^training/Adam/gradients/batch_normalization_2/moments/variance_grad/truediv*(
_output_shapes
:����������*
T0*B
_class8
64loc:@batch_normalization_2/moments/SquaredDifference
�
Rtraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/mul_1MulPtraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/MulPtraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/sub*
T0*B
_class8
64loc:@batch_normalization_2/moments/SquaredDifference*(
_output_shapes
:����������
�
Rtraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/ShapeShapedense_2/Relu*
T0*
out_type0*B
_class8
64loc:@batch_normalization_2/moments/SquaredDifference*
_output_shapes
:
�
Ttraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/Shape_1Shape*batch_normalization_2/moments/StopGradient*
T0*
out_type0*B
_class8
64loc:@batch_normalization_2/moments/SquaredDifference*
_output_shapes
:
�
btraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgsRtraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/ShapeTtraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/Shape_1*
T0*B
_class8
64loc:@batch_normalization_2/moments/SquaredDifference*2
_output_shapes 
:���������:���������
�
Ptraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/SumSumRtraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/mul_1btraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*B
_class8
64loc:@batch_normalization_2/moments/SquaredDifference*
_output_shapes
:
�
Ttraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/ReshapeReshapePtraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/SumRtraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/Shape*
T0*
Tshape0*B
_class8
64loc:@batch_normalization_2/moments/SquaredDifference*(
_output_shapes
:����������
�
Rtraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/Sum_1SumRtraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/mul_1dtraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/BroadcastGradientArgs:1*
T0*B
_class8
64loc:@batch_normalization_2/moments/SquaredDifference*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
Vtraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/Reshape_1ReshapeRtraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/Sum_1Ttraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/Shape_1*
T0*
Tshape0*B
_class8
64loc:@batch_normalization_2/moments/SquaredDifference*
_output_shapes
:	�
�
Ptraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/NegNegVtraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/Reshape_1*
T0*B
_class8
64loc:@batch_normalization_2/moments/SquaredDifference*
_output_shapes
:	�
�
Etraining/Adam/gradients/batch_normalization_2/moments/mean_grad/ShapeShapedense_2/Relu*
_output_shapes
:*
T0*
out_type0*5
_class+
)'loc:@batch_normalization_2/moments/mean
�
Dtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/SizeConst*
dtype0*
_output_shapes
: *
value	B :*5
_class+
)'loc:@batch_normalization_2/moments/mean
�
Ctraining/Adam/gradients/batch_normalization_2/moments/mean_grad/addAddV24batch_normalization_2/moments/mean/reduction_indicesDtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Size*
_output_shapes
:*
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean
�
Ctraining/Adam/gradients/batch_normalization_2/moments/mean_grad/modFloorModCtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/addDtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Size*
_output_shapes
:*
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean
�
Gtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Shape_1Const*
valueB:*5
_class+
)'loc:@batch_normalization_2/moments/mean*
dtype0*
_output_shapes
:
�
Ktraining/Adam/gradients/batch_normalization_2/moments/mean_grad/range/startConst*
value	B : *5
_class+
)'loc:@batch_normalization_2/moments/mean*
dtype0*
_output_shapes
: 
�
Ktraining/Adam/gradients/batch_normalization_2/moments/mean_grad/range/deltaConst*
value	B :*5
_class+
)'loc:@batch_normalization_2/moments/mean*
dtype0*
_output_shapes
: 
�
Etraining/Adam/gradients/batch_normalization_2/moments/mean_grad/rangeRangeKtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/range/startDtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/SizeKtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/range/delta*

Tidx0*5
_class+
)'loc:@batch_normalization_2/moments/mean*
_output_shapes
:
�
Jtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Fill/valueConst*
value	B :*5
_class+
)'loc:@batch_normalization_2/moments/mean*
dtype0*
_output_shapes
: 
�
Dtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/FillFillGtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Shape_1Jtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Fill/value*
_output_shapes
:*
T0*

index_type0*5
_class+
)'loc:@batch_normalization_2/moments/mean
�
Mtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/DynamicStitchDynamicStitchEtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/rangeCtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/modEtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/ShapeDtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Fill*
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean*
N*
_output_shapes
:
�
Itraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Maximum/yConst*
value	B :*5
_class+
)'loc:@batch_normalization_2/moments/mean*
dtype0*
_output_shapes
: 
�
Gtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/MaximumMaximumMtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/DynamicStitchItraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Maximum/y*
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean*
_output_shapes
:
�
Htraining/Adam/gradients/batch_normalization_2/moments/mean_grad/floordivFloorDivEtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/ShapeGtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Maximum*
_output_shapes
:*
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean
�
Gtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/ReshapeReshapeJtraining/Adam/gradients/batch_normalization_2/moments/Squeeze_grad/ReshapeMtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/DynamicStitch*
T0*
Tshape0*5
_class+
)'loc:@batch_normalization_2/moments/mean*0
_output_shapes
:������������������
�
Dtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/TileTileGtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/ReshapeHtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/floordiv*

Tmultiples0*
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean*0
_output_shapes
:������������������
�
Gtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Shape_2Shapedense_2/Relu*
_output_shapes
:*
T0*
out_type0*5
_class+
)'loc:@batch_normalization_2/moments/mean
�
Gtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Shape_3Const*
valueB"   �   *5
_class+
)'loc:@batch_normalization_2/moments/mean*
dtype0*
_output_shapes
:
�
Etraining/Adam/gradients/batch_normalization_2/moments/mean_grad/ConstConst*
valueB: *5
_class+
)'loc:@batch_normalization_2/moments/mean*
dtype0*
_output_shapes
:
�
Dtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/ProdProdGtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Shape_2Etraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Const*

Tidx0*
	keep_dims( *
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean*
_output_shapes
: 
�
Gtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Const_1Const*
valueB: *5
_class+
)'loc:@batch_normalization_2/moments/mean*
dtype0*
_output_shapes
:
�
Ftraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Prod_1ProdGtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Shape_3Gtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Const_1*

Tidx0*
	keep_dims( *
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean*
_output_shapes
: 
�
Ktraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Maximum_1/yConst*
value	B :*5
_class+
)'loc:@batch_normalization_2/moments/mean*
dtype0*
_output_shapes
: 
�
Itraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Maximum_1MaximumFtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Prod_1Ktraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Maximum_1/y*
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean*
_output_shapes
: 
�
Jtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/floordiv_1FloorDivDtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/ProdItraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Maximum_1*
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean*
_output_shapes
: 
�
Dtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/CastCastJtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/floordiv_1*

SrcT0*5
_class+
)'loc:@batch_normalization_2/moments/mean*
Truncate( *
_output_shapes
: *

DstT0
�
Gtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/truedivRealDivDtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/TileDtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Cast*(
_output_shapes
:����������*
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean
�
training/Adam/gradients/AddN_16AddNXtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1/Switch_grad/cond_gradJtraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/ReshapeTtraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/ReshapeGtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/truediv*
N*(
_output_shapes
:����������*
T0*
_class
loc:@dense_2/Relu
�
2training/Adam/gradients/dense_2/Relu_grad/ReluGradReluGradtraining/Adam/gradients/AddN_16dense_2/Relu*(
_output_shapes
:����������*
T0*
_class
loc:@dense_2/Relu
�
8training/Adam/gradients/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad2training/Adam/gradients/dense_2/Relu_grad/ReluGrad*
T0*"
_class
loc:@dense_2/BiasAdd*
data_formatNHWC*
_output_shapes	
:�
�
2training/Adam/gradients/dense_2/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_2/Relu_grad/ReluGraddense_2/MatMul/ReadVariableOp*
T0*!
_class
loc:@dense_2/MatMul*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
4training/Adam/gradients/dense_2/MatMul_grad/MatMul_1MatMul batch_normalization_1/cond/Merge2training/Adam/gradients/dense_2/Relu_grad/ReluGrad*
T0*!
_class
loc:@dense_2/MatMul* 
_output_shapes
:
��*
transpose_a(*
transpose_b( 
�
Gtraining/Adam/gradients/batch_normalization_1/cond/Merge_grad/cond_gradSwitch2training/Adam/gradients/dense_2/MatMul_grad/MatMul"batch_normalization_1/cond/pred_id*
T0*!
_class
loc:@dense_2/MatMul*<
_output_shapes*
(:����������:����������
�
Mtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/ShapeShape*batch_normalization_1/cond/batchnorm/mul_1*
T0*
out_type0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/add_1*
_output_shapes
:
�
Otraining/Adam/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/Shape_1Shape(batch_normalization_1/cond/batchnorm/sub*
T0*
out_type0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/add_1*
_output_shapes
:
�
]training/Adam/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsMtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/ShapeOtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/Shape_1*
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/add_1*2
_output_shapes 
:���������:���������
�
Ktraining/Adam/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/SumSumGtraining/Adam/gradients/batch_normalization_1/cond/Merge_grad/cond_grad]training/Adam/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/BroadcastGradientArgs*
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/add_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
Otraining/Adam/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/ReshapeReshapeKtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/SumMtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/Shape*
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/add_1*(
_output_shapes
:����������
�
Mtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/Sum_1SumGtraining/Adam/gradients/batch_normalization_1/cond/Merge_grad/cond_grad_training/Adam/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/add_1
�
Qtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/Reshape_1ReshapeMtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/Sum_1Otraining/Adam/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/Shape_1*
_output_shapes	
:�*
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/add_1
�
!training/Adam/gradients/Switch_16Switch%batch_normalization_1/batchnorm/add_1"batch_normalization_1/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1*<
_output_shapes*
(:����������:����������
�
#training/Adam/gradients/Identity_16Identity!training/Adam/gradients/Switch_16*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1*(
_output_shapes
:����������
�
 training/Adam/gradients/Shape_11Shape!training/Adam/gradients/Switch_16*
_output_shapes
:*
T0*
out_type0*8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1
�
&training/Adam/gradients/zeros_16/ConstConst$^training/Adam/gradients/Identity_16*
valueB
 *    *8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1*
dtype0*
_output_shapes
: 
�
 training/Adam/gradients/zeros_16Fill training/Adam/gradients/Shape_11&training/Adam/gradients/zeros_16/Const*
T0*

index_type0*8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1*(
_output_shapes
:����������
�
Jtraining/Adam/gradients/batch_normalization_1/cond/Switch_1_grad/cond_gradMerge training/Adam/gradients/zeros_16Itraining/Adam/gradients/batch_normalization_1/cond/Merge_grad/cond_grad:1*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1*
N**
_output_shapes
:����������: 
�
Mtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/ShapeShape1batch_normalization_1/cond/batchnorm/mul_1/Switch*
T0*
out_type0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/mul_1*
_output_shapes
:
�
Otraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/Shape_1Shape(batch_normalization_1/cond/batchnorm/mul*
T0*
out_type0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/mul_1*
_output_shapes
:
�
]training/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsMtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/ShapeOtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/Shape_1*
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/mul_1*2
_output_shapes 
:���������:���������
�
Ktraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/MulMulOtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/Reshape(batch_normalization_1/cond/batchnorm/mul*
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/mul_1*(
_output_shapes
:����������
�
Ktraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/SumSumKtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/Mul]training/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/mul_1
�
Otraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/ReshapeReshapeKtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/SumMtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/mul_1*(
_output_shapes
:����������
�
Mtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/Mul_1Mul1batch_normalization_1/cond/batchnorm/mul_1/SwitchOtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/Reshape*(
_output_shapes
:����������*
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/mul_1
�
Mtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/Sum_1SumMtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/Mul_1_training/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/mul_1*
_output_shapes
:
�
Qtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/Reshape_1ReshapeMtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/Sum_1Otraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/mul_1*
_output_shapes	
:�
�
Itraining/Adam/gradients/batch_normalization_1/cond/batchnorm/sub_grad/NegNegQtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:�*
T0*;
_class1
/-loc:@batch_normalization_1/cond/batchnorm/sub
�
Htraining/Adam/gradients/batch_normalization_1/batchnorm/add_1_grad/ShapeShape%batch_normalization_1/batchnorm/mul_1*
_output_shapes
:*
T0*
out_type0*8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1
�
Jtraining/Adam/gradients/batch_normalization_1/batchnorm/add_1_grad/Shape_1Shape#batch_normalization_1/batchnorm/sub*
_output_shapes
:*
T0*
out_type0*8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1
�
Xtraining/Adam/gradients/batch_normalization_1/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsHtraining/Adam/gradients/batch_normalization_1/batchnorm/add_1_grad/ShapeJtraining/Adam/gradients/batch_normalization_1/batchnorm/add_1_grad/Shape_1*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1*2
_output_shapes 
:���������:���������
�
Ftraining/Adam/gradients/batch_normalization_1/batchnorm/add_1_grad/SumSumJtraining/Adam/gradients/batch_normalization_1/cond/Switch_1_grad/cond_gradXtraining/Adam/gradients/batch_normalization_1/batchnorm/add_1_grad/BroadcastGradientArgs*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
Jtraining/Adam/gradients/batch_normalization_1/batchnorm/add_1_grad/ReshapeReshapeFtraining/Adam/gradients/batch_normalization_1/batchnorm/add_1_grad/SumHtraining/Adam/gradients/batch_normalization_1/batchnorm/add_1_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1
�
Htraining/Adam/gradients/batch_normalization_1/batchnorm/add_1_grad/Sum_1SumJtraining/Adam/gradients/batch_normalization_1/cond/Switch_1_grad/cond_gradZtraining/Adam/gradients/batch_normalization_1/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
Ltraining/Adam/gradients/batch_normalization_1/batchnorm/add_1_grad/Reshape_1ReshapeHtraining/Adam/gradients/batch_normalization_1/batchnorm/add_1_grad/Sum_1Jtraining/Adam/gradients/batch_normalization_1/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1*
_output_shapes	
:�
�
!training/Adam/gradients/Switch_17Switchdense_1/Relu"batch_normalization_1/cond/pred_id*
T0*
_class
loc:@dense_1/Relu*<
_output_shapes*
(:����������:����������
�
#training/Adam/gradients/Identity_17Identity#training/Adam/gradients/Switch_17:1*(
_output_shapes
:����������*
T0*
_class
loc:@dense_1/Relu
�
 training/Adam/gradients/Shape_12Shape#training/Adam/gradients/Switch_17:1*
T0*
out_type0*
_class
loc:@dense_1/Relu*
_output_shapes
:
�
&training/Adam/gradients/zeros_17/ConstConst$^training/Adam/gradients/Identity_17*
valueB
 *    *
_class
loc:@dense_1/Relu*
dtype0*
_output_shapes
: 
�
 training/Adam/gradients/zeros_17Fill training/Adam/gradients/Shape_12&training/Adam/gradients/zeros_17/Const*
T0*

index_type0*
_class
loc:@dense_1/Relu*(
_output_shapes
:����������
�
Xtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1/Switch_grad/cond_gradMergeOtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/Reshape training/Adam/gradients/zeros_17*
N**
_output_shapes
:����������: *
T0*
_class
loc:@dense_1/Relu
�
Ktraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_2_grad/MulMulItraining/Adam/gradients/batch_normalization_1/cond/batchnorm/sub_grad/Neg(batch_normalization_1/cond/batchnorm/mul*
_output_shapes	
:�*
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/mul_2
�
Mtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_2_grad/Mul_1MulItraining/Adam/gradients/batch_normalization_1/cond/batchnorm/sub_grad/Neg5batch_normalization_1/cond/batchnorm/ReadVariableOp_1*
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/mul_2*
_output_shapes	
:�
�
Htraining/Adam/gradients/batch_normalization_1/batchnorm/mul_1_grad/ShapeShapedense_1/Relu*
_output_shapes
:*
T0*
out_type0*8
_class.
,*loc:@batch_normalization_1/batchnorm/mul_1
�
Jtraining/Adam/gradients/batch_normalization_1/batchnorm/mul_1_grad/Shape_1Shape#batch_normalization_1/batchnorm/mul*
T0*
out_type0*8
_class.
,*loc:@batch_normalization_1/batchnorm/mul_1*
_output_shapes
:
�
Xtraining/Adam/gradients/batch_normalization_1/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsHtraining/Adam/gradients/batch_normalization_1/batchnorm/mul_1_grad/ShapeJtraining/Adam/gradients/batch_normalization_1/batchnorm/mul_1_grad/Shape_1*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/mul_1*2
_output_shapes 
:���������:���������
�
Ftraining/Adam/gradients/batch_normalization_1/batchnorm/mul_1_grad/MulMulJtraining/Adam/gradients/batch_normalization_1/batchnorm/add_1_grad/Reshape#batch_normalization_1/batchnorm/mul*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/mul_1*(
_output_shapes
:����������
�
Ftraining/Adam/gradients/batch_normalization_1/batchnorm/mul_1_grad/SumSumFtraining/Adam/gradients/batch_normalization_1/batchnorm/mul_1_grad/MulXtraining/Adam/gradients/batch_normalization_1/batchnorm/mul_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/mul_1*
_output_shapes
:
�
Jtraining/Adam/gradients/batch_normalization_1/batchnorm/mul_1_grad/ReshapeReshapeFtraining/Adam/gradients/batch_normalization_1/batchnorm/mul_1_grad/SumHtraining/Adam/gradients/batch_normalization_1/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_1/batchnorm/mul_1*(
_output_shapes
:����������
�
Htraining/Adam/gradients/batch_normalization_1/batchnorm/mul_1_grad/Mul_1Muldense_1/ReluJtraining/Adam/gradients/batch_normalization_1/batchnorm/add_1_grad/Reshape*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/mul_1*(
_output_shapes
:����������
�
Htraining/Adam/gradients/batch_normalization_1/batchnorm/mul_1_grad/Sum_1SumHtraining/Adam/gradients/batch_normalization_1/batchnorm/mul_1_grad/Mul_1Ztraining/Adam/gradients/batch_normalization_1/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/mul_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
Ltraining/Adam/gradients/batch_normalization_1/batchnorm/mul_1_grad/Reshape_1ReshapeHtraining/Adam/gradients/batch_normalization_1/batchnorm/mul_1_grad/Sum_1Jtraining/Adam/gradients/batch_normalization_1/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_1/batchnorm/mul_1*
_output_shapes	
:�
�
Dtraining/Adam/gradients/batch_normalization_1/batchnorm/sub_grad/NegNegLtraining/Adam/gradients/batch_normalization_1/batchnorm/add_1_grad/Reshape_1*
T0*6
_class,
*(loc:@batch_normalization_1/batchnorm/sub*
_output_shapes	
:�
�
!training/Adam/gradients/Switch_18Switchbatch_normalization_1/beta"batch_normalization_1/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_1/beta*
_output_shapes
: : 
�
#training/Adam/gradients/Identity_18Identity#training/Adam/gradients/Switch_18:1*
T0*-
_class#
!loc:@batch_normalization_1/beta*
_output_shapes
: 
�
'training/Adam/gradients/VariableShape_6VariableShape#training/Adam/gradients/Switch_18:1$^training/Adam/gradients/Identity_18*
out_type0*-
_class#
!loc:@batch_normalization_1/beta*
_output_shapes
:
�
&training/Adam/gradients/zeros_18/ConstConst$^training/Adam/gradients/Identity_18*
valueB
 *    *-
_class#
!loc:@batch_normalization_1/beta*
dtype0*
_output_shapes
: 
�
 training/Adam/gradients/zeros_18Fill'training/Adam/gradients/VariableShape_6&training/Adam/gradients/zeros_18/Const*
T0*

index_type0*-
_class#
!loc:@batch_normalization_1/beta*#
_output_shapes
:���������
�
ctraining/Adam/gradients/batch_normalization_1/cond/batchnorm/ReadVariableOp_2/Switch_grad/cond_gradMergeQtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/Reshape_1 training/Adam/gradients/zeros_18*
T0*-
_class#
!loc:@batch_normalization_1/beta*
N*%
_output_shapes
:���������: 
�
training/Adam/gradients/AddN_17AddNQtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/Reshape_1Mtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_2_grad/Mul_1*
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/mul_1*
N*
_output_shapes	
:�
�
Itraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_grad/MulMultraining/Adam/gradients/AddN_177batch_normalization_1/cond/batchnorm/mul/ReadVariableOp*
T0*;
_class1
/-loc:@batch_normalization_1/cond/batchnorm/mul*
_output_shapes	
:�
�
Ktraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_grad/Mul_1Multraining/Adam/gradients/AddN_17*batch_normalization_1/cond/batchnorm/Rsqrt*
_output_shapes	
:�*
T0*;
_class1
/-loc:@batch_normalization_1/cond/batchnorm/mul
�
Ftraining/Adam/gradients/batch_normalization_1/batchnorm/mul_2_grad/MulMulDtraining/Adam/gradients/batch_normalization_1/batchnorm/sub_grad/Neg#batch_normalization_1/batchnorm/mul*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/mul_2*
_output_shapes	
:�
�
Htraining/Adam/gradients/batch_normalization_1/batchnorm/mul_2_grad/Mul_1MulDtraining/Adam/gradients/batch_normalization_1/batchnorm/sub_grad/Neg%batch_normalization_1/moments/Squeeze*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/mul_2*
_output_shapes	
:�
�
training/Adam/gradients/AddN_18AddNctraining/Adam/gradients/batch_normalization_1/cond/batchnorm/ReadVariableOp_2/Switch_grad/cond_gradLtraining/Adam/gradients/batch_normalization_1/batchnorm/add_1_grad/Reshape_1*
T0*-
_class#
!loc:@batch_normalization_1/beta*
N*
_output_shapes	
:�
�
Htraining/Adam/gradients/batch_normalization_1/moments/Squeeze_grad/ShapeConst*
valueB"   �   *8
_class.
,*loc:@batch_normalization_1/moments/Squeeze*
dtype0*
_output_shapes
:
�
Jtraining/Adam/gradients/batch_normalization_1/moments/Squeeze_grad/ReshapeReshapeFtraining/Adam/gradients/batch_normalization_1/batchnorm/mul_2_grad/MulHtraining/Adam/gradients/batch_normalization_1/moments/Squeeze_grad/Shape*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_1/moments/Squeeze*
_output_shapes
:	�
�
training/Adam/gradients/AddN_19AddNLtraining/Adam/gradients/batch_normalization_1/batchnorm/mul_1_grad/Reshape_1Htraining/Adam/gradients/batch_normalization_1/batchnorm/mul_2_grad/Mul_1*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/mul_1*
N*
_output_shapes	
:�
�
Dtraining/Adam/gradients/batch_normalization_1/batchnorm/mul_grad/MulMultraining/Adam/gradients/AddN_192batch_normalization_1/batchnorm/mul/ReadVariableOp*
T0*6
_class,
*(loc:@batch_normalization_1/batchnorm/mul*
_output_shapes	
:�
�
Ftraining/Adam/gradients/batch_normalization_1/batchnorm/mul_grad/Mul_1Multraining/Adam/gradients/AddN_19%batch_normalization_1/batchnorm/Rsqrt*
T0*6
_class,
*(loc:@batch_normalization_1/batchnorm/mul*
_output_shapes	
:�
�
!training/Adam/gradients/Switch_19Switchbatch_normalization_1/gamma"batch_normalization_1/cond/pred_id*
_output_shapes
: : *
T0*.
_class$
" loc:@batch_normalization_1/gamma
�
#training/Adam/gradients/Identity_19Identity#training/Adam/gradients/Switch_19:1*
T0*.
_class$
" loc:@batch_normalization_1/gamma*
_output_shapes
: 
�
'training/Adam/gradients/VariableShape_7VariableShape#training/Adam/gradients/Switch_19:1$^training/Adam/gradients/Identity_19*
out_type0*.
_class$
" loc:@batch_normalization_1/gamma*
_output_shapes
:
�
&training/Adam/gradients/zeros_19/ConstConst$^training/Adam/gradients/Identity_19*
dtype0*
_output_shapes
: *
valueB
 *    *.
_class$
" loc:@batch_normalization_1/gamma
�
 training/Adam/gradients/zeros_19Fill'training/Adam/gradients/VariableShape_7&training/Adam/gradients/zeros_19/Const*#
_output_shapes
:���������*
T0*

index_type0*.
_class$
" loc:@batch_normalization_1/gamma
�
etraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul/ReadVariableOp/Switch_grad/cond_gradMergeKtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_grad/Mul_1 training/Adam/gradients/zeros_19*
T0*.
_class$
" loc:@batch_normalization_1/gamma*
N*%
_output_shapes
:���������: 
�
Ltraining/Adam/gradients/batch_normalization_1/batchnorm/Rsqrt_grad/RsqrtGrad	RsqrtGrad%batch_normalization_1/batchnorm/RsqrtDtraining/Adam/gradients/batch_normalization_1/batchnorm/mul_grad/Mul*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/Rsqrt*
_output_shapes	
:�
�
Vtraining/Adam/gradients/batch_normalization_1/batchnorm/add_grad/Sum/reduction_indicesConst*
valueB: *6
_class,
*(loc:@batch_normalization_1/batchnorm/add*
dtype0*
_output_shapes
:
�
Dtraining/Adam/gradients/batch_normalization_1/batchnorm/add_grad/SumSumLtraining/Adam/gradients/batch_normalization_1/batchnorm/Rsqrt_grad/RsqrtGradVtraining/Adam/gradients/batch_normalization_1/batchnorm/add_grad/Sum/reduction_indices*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0*6
_class,
*(loc:@batch_normalization_1/batchnorm/add
�
Ntraining/Adam/gradients/batch_normalization_1/batchnorm/add_grad/Reshape/shapeConst*
valueB *6
_class,
*(loc:@batch_normalization_1/batchnorm/add*
dtype0*
_output_shapes
: 
�
Htraining/Adam/gradients/batch_normalization_1/batchnorm/add_grad/ReshapeReshapeDtraining/Adam/gradients/batch_normalization_1/batchnorm/add_grad/SumNtraining/Adam/gradients/batch_normalization_1/batchnorm/add_grad/Reshape/shape*
T0*
Tshape0*6
_class,
*(loc:@batch_normalization_1/batchnorm/add*
_output_shapes
: 
�
training/Adam/gradients/AddN_20AddNetraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul/ReadVariableOp/Switch_grad/cond_gradFtraining/Adam/gradients/batch_normalization_1/batchnorm/mul_grad/Mul_1*
N*
_output_shapes	
:�*
T0*.
_class$
" loc:@batch_normalization_1/gamma
�
Jtraining/Adam/gradients/batch_normalization_1/moments/Squeeze_1_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"   �   *:
_class0
.,loc:@batch_normalization_1/moments/Squeeze_1
�
Ltraining/Adam/gradients/batch_normalization_1/moments/Squeeze_1_grad/ReshapeReshapeLtraining/Adam/gradients/batch_normalization_1/batchnorm/Rsqrt_grad/RsqrtGradJtraining/Adam/gradients/batch_normalization_1/moments/Squeeze_1_grad/Shape*
T0*
Tshape0*:
_class0
.,loc:@batch_normalization_1/moments/Squeeze_1*
_output_shapes
:	�
�
Itraining/Adam/gradients/batch_normalization_1/moments/variance_grad/ShapeShape/batch_normalization_1/moments/SquaredDifference*
T0*
out_type0*9
_class/
-+loc:@batch_normalization_1/moments/variance*
_output_shapes
:
�
Htraining/Adam/gradients/batch_normalization_1/moments/variance_grad/SizeConst*
value	B :*9
_class/
-+loc:@batch_normalization_1/moments/variance*
dtype0*
_output_shapes
: 
�
Gtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/addAddV28batch_normalization_1/moments/variance/reduction_indicesHtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Size*
T0*9
_class/
-+loc:@batch_normalization_1/moments/variance*
_output_shapes
:
�
Gtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/modFloorModGtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/addHtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Size*
T0*9
_class/
-+loc:@batch_normalization_1/moments/variance*
_output_shapes
:
�
Ktraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Shape_1Const*
valueB:*9
_class/
-+loc:@batch_normalization_1/moments/variance*
dtype0*
_output_shapes
:
�
Otraining/Adam/gradients/batch_normalization_1/moments/variance_grad/range/startConst*
value	B : *9
_class/
-+loc:@batch_normalization_1/moments/variance*
dtype0*
_output_shapes
: 
�
Otraining/Adam/gradients/batch_normalization_1/moments/variance_grad/range/deltaConst*
value	B :*9
_class/
-+loc:@batch_normalization_1/moments/variance*
dtype0*
_output_shapes
: 
�
Itraining/Adam/gradients/batch_normalization_1/moments/variance_grad/rangeRangeOtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/range/startHtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/SizeOtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/range/delta*
_output_shapes
:*

Tidx0*9
_class/
-+loc:@batch_normalization_1/moments/variance
�
Ntraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Fill/valueConst*
value	B :*9
_class/
-+loc:@batch_normalization_1/moments/variance*
dtype0*
_output_shapes
: 
�
Htraining/Adam/gradients/batch_normalization_1/moments/variance_grad/FillFillKtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Shape_1Ntraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Fill/value*
T0*

index_type0*9
_class/
-+loc:@batch_normalization_1/moments/variance*
_output_shapes
:
�
Qtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/DynamicStitchDynamicStitchItraining/Adam/gradients/batch_normalization_1/moments/variance_grad/rangeGtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/modItraining/Adam/gradients/batch_normalization_1/moments/variance_grad/ShapeHtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Fill*
T0*9
_class/
-+loc:@batch_normalization_1/moments/variance*
N*
_output_shapes
:
�
Mtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :*9
_class/
-+loc:@batch_normalization_1/moments/variance
�
Ktraining/Adam/gradients/batch_normalization_1/moments/variance_grad/MaximumMaximumQtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/DynamicStitchMtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Maximum/y*
T0*9
_class/
-+loc:@batch_normalization_1/moments/variance*
_output_shapes
:
�
Ltraining/Adam/gradients/batch_normalization_1/moments/variance_grad/floordivFloorDivItraining/Adam/gradients/batch_normalization_1/moments/variance_grad/ShapeKtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Maximum*
_output_shapes
:*
T0*9
_class/
-+loc:@batch_normalization_1/moments/variance
�
Ktraining/Adam/gradients/batch_normalization_1/moments/variance_grad/ReshapeReshapeLtraining/Adam/gradients/batch_normalization_1/moments/Squeeze_1_grad/ReshapeQtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/DynamicStitch*0
_output_shapes
:������������������*
T0*
Tshape0*9
_class/
-+loc:@batch_normalization_1/moments/variance
�
Htraining/Adam/gradients/batch_normalization_1/moments/variance_grad/TileTileKtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/ReshapeLtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/floordiv*
T0*9
_class/
-+loc:@batch_normalization_1/moments/variance*0
_output_shapes
:������������������*

Tmultiples0
�
Ktraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Shape_2Shape/batch_normalization_1/moments/SquaredDifference*
_output_shapes
:*
T0*
out_type0*9
_class/
-+loc:@batch_normalization_1/moments/variance
�
Ktraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Shape_3Const*
valueB"   �   *9
_class/
-+loc:@batch_normalization_1/moments/variance*
dtype0*
_output_shapes
:
�
Itraining/Adam/gradients/batch_normalization_1/moments/variance_grad/ConstConst*
valueB: *9
_class/
-+loc:@batch_normalization_1/moments/variance*
dtype0*
_output_shapes
:
�
Htraining/Adam/gradients/batch_normalization_1/moments/variance_grad/ProdProdKtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Shape_2Itraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0*9
_class/
-+loc:@batch_normalization_1/moments/variance
�
Ktraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Const_1Const*
valueB: *9
_class/
-+loc:@batch_normalization_1/moments/variance*
dtype0*
_output_shapes
:
�
Jtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Prod_1ProdKtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Shape_3Ktraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Const_1*
T0*9
_class/
-+loc:@batch_normalization_1/moments/variance*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
Otraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Maximum_1/yConst*
value	B :*9
_class/
-+loc:@batch_normalization_1/moments/variance*
dtype0*
_output_shapes
: 
�
Mtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Maximum_1MaximumJtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Prod_1Otraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Maximum_1/y*
T0*9
_class/
-+loc:@batch_normalization_1/moments/variance*
_output_shapes
: 
�
Ntraining/Adam/gradients/batch_normalization_1/moments/variance_grad/floordiv_1FloorDivHtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/ProdMtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Maximum_1*
T0*9
_class/
-+loc:@batch_normalization_1/moments/variance*
_output_shapes
: 
�
Htraining/Adam/gradients/batch_normalization_1/moments/variance_grad/CastCastNtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/floordiv_1*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0*9
_class/
-+loc:@batch_normalization_1/moments/variance
�
Ktraining/Adam/gradients/batch_normalization_1/moments/variance_grad/truedivRealDivHtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/TileHtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Cast*
T0*9
_class/
-+loc:@batch_normalization_1/moments/variance*(
_output_shapes
:����������
�
Straining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/scalarConstL^training/Adam/gradients/batch_normalization_1/moments/variance_grad/truediv*
valueB
 *   @*B
_class8
64loc:@batch_normalization_1/moments/SquaredDifference*
dtype0*
_output_shapes
: 
�
Ptraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/MulMulStraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/scalarKtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/truediv*
T0*B
_class8
64loc:@batch_normalization_1/moments/SquaredDifference*(
_output_shapes
:����������
�
Ptraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/subSubdense_1/Relu*batch_normalization_1/moments/StopGradientL^training/Adam/gradients/batch_normalization_1/moments/variance_grad/truediv*
T0*B
_class8
64loc:@batch_normalization_1/moments/SquaredDifference*(
_output_shapes
:����������
�
Rtraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/mul_1MulPtraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/MulPtraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/sub*
T0*B
_class8
64loc:@batch_normalization_1/moments/SquaredDifference*(
_output_shapes
:����������
�
Rtraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/ShapeShapedense_1/Relu*
T0*
out_type0*B
_class8
64loc:@batch_normalization_1/moments/SquaredDifference*
_output_shapes
:
�
Ttraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/Shape_1Shape*batch_normalization_1/moments/StopGradient*
_output_shapes
:*
T0*
out_type0*B
_class8
64loc:@batch_normalization_1/moments/SquaredDifference
�
btraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgsRtraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/ShapeTtraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/Shape_1*
T0*B
_class8
64loc:@batch_normalization_1/moments/SquaredDifference*2
_output_shapes 
:���������:���������
�
Ptraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/SumSumRtraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/mul_1btraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*B
_class8
64loc:@batch_normalization_1/moments/SquaredDifference*
_output_shapes
:
�
Ttraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/ReshapeReshapePtraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/SumRtraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/Shape*
T0*
Tshape0*B
_class8
64loc:@batch_normalization_1/moments/SquaredDifference*(
_output_shapes
:����������
�
Rtraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/Sum_1SumRtraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/mul_1dtraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/BroadcastGradientArgs:1*
T0*B
_class8
64loc:@batch_normalization_1/moments/SquaredDifference*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
Vtraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/Reshape_1ReshapeRtraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/Sum_1Ttraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/Shape_1*
T0*
Tshape0*B
_class8
64loc:@batch_normalization_1/moments/SquaredDifference*
_output_shapes
:	�
�
Ptraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/NegNegVtraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/Reshape_1*
T0*B
_class8
64loc:@batch_normalization_1/moments/SquaredDifference*
_output_shapes
:	�
�
Etraining/Adam/gradients/batch_normalization_1/moments/mean_grad/ShapeShapedense_1/Relu*
T0*
out_type0*5
_class+
)'loc:@batch_normalization_1/moments/mean*
_output_shapes
:
�
Dtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/SizeConst*
value	B :*5
_class+
)'loc:@batch_normalization_1/moments/mean*
dtype0*
_output_shapes
: 
�
Ctraining/Adam/gradients/batch_normalization_1/moments/mean_grad/addAddV24batch_normalization_1/moments/mean/reduction_indicesDtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Size*
T0*5
_class+
)'loc:@batch_normalization_1/moments/mean*
_output_shapes
:
�
Ctraining/Adam/gradients/batch_normalization_1/moments/mean_grad/modFloorModCtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/addDtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Size*
T0*5
_class+
)'loc:@batch_normalization_1/moments/mean*
_output_shapes
:
�
Gtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:*5
_class+
)'loc:@batch_normalization_1/moments/mean
�
Ktraining/Adam/gradients/batch_normalization_1/moments/mean_grad/range/startConst*
dtype0*
_output_shapes
: *
value	B : *5
_class+
)'loc:@batch_normalization_1/moments/mean
�
Ktraining/Adam/gradients/batch_normalization_1/moments/mean_grad/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :*5
_class+
)'loc:@batch_normalization_1/moments/mean
�
Etraining/Adam/gradients/batch_normalization_1/moments/mean_grad/rangeRangeKtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/range/startDtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/SizeKtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/range/delta*5
_class+
)'loc:@batch_normalization_1/moments/mean*
_output_shapes
:*

Tidx0
�
Jtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Fill/valueConst*
value	B :*5
_class+
)'loc:@batch_normalization_1/moments/mean*
dtype0*
_output_shapes
: 
�
Dtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/FillFillGtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Shape_1Jtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Fill/value*
_output_shapes
:*
T0*

index_type0*5
_class+
)'loc:@batch_normalization_1/moments/mean
�
Mtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/DynamicStitchDynamicStitchEtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/rangeCtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/modEtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/ShapeDtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Fill*
T0*5
_class+
)'loc:@batch_normalization_1/moments/mean*
N*
_output_shapes
:
�
Itraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Maximum/yConst*
value	B :*5
_class+
)'loc:@batch_normalization_1/moments/mean*
dtype0*
_output_shapes
: 
�
Gtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/MaximumMaximumMtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/DynamicStitchItraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Maximum/y*
T0*5
_class+
)'loc:@batch_normalization_1/moments/mean*
_output_shapes
:
�
Htraining/Adam/gradients/batch_normalization_1/moments/mean_grad/floordivFloorDivEtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/ShapeGtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Maximum*
T0*5
_class+
)'loc:@batch_normalization_1/moments/mean*
_output_shapes
:
�
Gtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/ReshapeReshapeJtraining/Adam/gradients/batch_normalization_1/moments/Squeeze_grad/ReshapeMtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/DynamicStitch*
T0*
Tshape0*5
_class+
)'loc:@batch_normalization_1/moments/mean*0
_output_shapes
:������������������
�
Dtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/TileTileGtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/ReshapeHtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/floordiv*

Tmultiples0*
T0*5
_class+
)'loc:@batch_normalization_1/moments/mean*0
_output_shapes
:������������������
�
Gtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Shape_2Shapedense_1/Relu*
T0*
out_type0*5
_class+
)'loc:@batch_normalization_1/moments/mean*
_output_shapes
:
�
Gtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Shape_3Const*
valueB"   �   *5
_class+
)'loc:@batch_normalization_1/moments/mean*
dtype0*
_output_shapes
:
�
Etraining/Adam/gradients/batch_normalization_1/moments/mean_grad/ConstConst*
valueB: *5
_class+
)'loc:@batch_normalization_1/moments/mean*
dtype0*
_output_shapes
:
�
Dtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/ProdProdGtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Shape_2Etraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0*5
_class+
)'loc:@batch_normalization_1/moments/mean
�
Gtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: *5
_class+
)'loc:@batch_normalization_1/moments/mean
�
Ftraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Prod_1ProdGtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Shape_3Gtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Const_1*

Tidx0*
	keep_dims( *
T0*5
_class+
)'loc:@batch_normalization_1/moments/mean*
_output_shapes
: 
�
Ktraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Maximum_1/yConst*
value	B :*5
_class+
)'loc:@batch_normalization_1/moments/mean*
dtype0*
_output_shapes
: 
�
Itraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Maximum_1MaximumFtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Prod_1Ktraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Maximum_1/y*
T0*5
_class+
)'loc:@batch_normalization_1/moments/mean*
_output_shapes
: 
�
Jtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/floordiv_1FloorDivDtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/ProdItraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Maximum_1*
T0*5
_class+
)'loc:@batch_normalization_1/moments/mean*
_output_shapes
: 
�
Dtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/CastCastJtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/floordiv_1*

SrcT0*5
_class+
)'loc:@batch_normalization_1/moments/mean*
Truncate( *
_output_shapes
: *

DstT0
�
Gtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/truedivRealDivDtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/TileDtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Cast*
T0*5
_class+
)'loc:@batch_normalization_1/moments/mean*(
_output_shapes
:����������
�
training/Adam/gradients/AddN_21AddNXtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1/Switch_grad/cond_gradJtraining/Adam/gradients/batch_normalization_1/batchnorm/mul_1_grad/ReshapeTtraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/ReshapeGtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/truediv*
T0*
_class
loc:@dense_1/Relu*
N*(
_output_shapes
:����������
�
2training/Adam/gradients/dense_1/Relu_grad/ReluGradReluGradtraining/Adam/gradients/AddN_21dense_1/Relu*
T0*
_class
loc:@dense_1/Relu*(
_output_shapes
:����������
�
8training/Adam/gradients/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad2training/Adam/gradients/dense_1/Relu_grad/ReluGrad*
T0*"
_class
loc:@dense_1/BiasAdd*
data_formatNHWC*
_output_shapes	
:�
�
2training/Adam/gradients/dense_1/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_1/Relu_grad/ReluGraddense_1/MatMul/ReadVariableOp*(
_output_shapes
:����������*
transpose_a( *
transpose_b(*
T0*!
_class
loc:@dense_1/MatMul
�
4training/Adam/gradients/dense_1/MatMul_grad/MatMul_1MatMuldense_1_input2training/Adam/gradients/dense_1/Relu_grad/ReluGrad*
T0*!
_class
loc:@dense_1/MatMul* 
_output_shapes
:
��*
transpose_a(*
transpose_b( 
U
training/Adam/ConstConst*
value	B	 R*
dtype0	*
_output_shapes
: 
k
!training/Adam/AssignAddVariableOpAssignAddVariableOpAdam/iterationstraining/Adam/Const*
dtype0	
�
training/Adam/ReadVariableOpReadVariableOpAdam/iterations"^training/Adam/AssignAddVariableOp*
dtype0	*
_output_shapes
: 
i
!training/Adam/Cast/ReadVariableOpReadVariableOpAdam/iterations*
dtype0	*
_output_shapes
: 
}
training/Adam/CastCast!training/Adam/Cast/ReadVariableOp*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0	
X
training/Adam/add/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
d
training/Adam/addAddV2training/Adam/Casttraining/Adam/add/y*
T0*
_output_shapes
: 
d
 training/Adam/Pow/ReadVariableOpReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
n
training/Adam/PowPow training/Adam/Pow/ReadVariableOptraining/Adam/add*
_output_shapes
: *
T0
X
training/Adam/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
a
training/Adam/subSubtraining/Adam/sub/xtraining/Adam/Pow*
_output_shapes
: *
T0
Z
training/Adam/Const_1Const*
dtype0*
_output_shapes
: *
valueB
 *    
Z
training/Adam/Const_2Const*
valueB
 *  �*
dtype0*
_output_shapes
: 
y
#training/Adam/clip_by_value/MinimumMinimumtraining/Adam/subtraining/Adam/Const_2*
T0*
_output_shapes
: 
�
training/Adam/clip_by_valueMaximum#training/Adam/clip_by_value/Minimumtraining/Adam/Const_1*
T0*
_output_shapes
: 
X
training/Adam/SqrtSqrttraining/Adam/clip_by_value*
T0*
_output_shapes
: 
f
"training/Adam/Pow_1/ReadVariableOpReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
r
training/Adam/Pow_1Pow"training/Adam/Pow_1/ReadVariableOptraining/Adam/add*
T0*
_output_shapes
: 
Z
training/Adam/sub_1/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
g
training/Adam/sub_1Subtraining/Adam/sub_1/xtraining/Adam/Pow_1*
T0*
_output_shapes
: 
j
training/Adam/truedivRealDivtraining/Adam/Sqrttraining/Adam/sub_1*
T0*
_output_shapes
: 
i
training/Adam/ReadVariableOp_1ReadVariableOpAdam/learning_rate*
dtype0*
_output_shapes
: 
p
training/Adam/mulMultraining/Adam/ReadVariableOp_1training/Adam/truediv*
T0*
_output_shapes
: 
r
!training/Adam/m_0/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB"   �   
\
training/Adam/m_0/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/m_0Fill!training/Adam/m_0/shape_as_tensortraining/Adam/m_0/Const*
T0*

index_type0* 
_output_shapes
:
��
�
training/Adam/m_0_1VarHandleOp*$
shared_nametraining/Adam/m_0_1*&
_class
loc:@training/Adam/m_0_1*
	container *
shape:
��*
dtype0*
_output_shapes
: 
w
4training/Adam/m_0_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/m_0_1*
_output_shapes
: 
c
training/Adam/m_0_1/AssignAssignVariableOptraining/Adam/m_0_1training/Adam/m_0*
dtype0
}
'training/Adam/m_0_1/Read/ReadVariableOpReadVariableOptraining/Adam/m_0_1*
dtype0* 
_output_shapes
:
��
`
training/Adam/m_1Const*
valueB�*    *
dtype0*
_output_shapes	
:�
�
training/Adam/m_1_1VarHandleOp*$
shared_nametraining/Adam/m_1_1*&
_class
loc:@training/Adam/m_1_1*
	container *
shape:�*
dtype0*
_output_shapes
: 
w
4training/Adam/m_1_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/m_1_1*
_output_shapes
: 
c
training/Adam/m_1_1/AssignAssignVariableOptraining/Adam/m_1_1training/Adam/m_1*
dtype0
x
'training/Adam/m_1_1/Read/ReadVariableOpReadVariableOptraining/Adam/m_1_1*
dtype0*
_output_shapes	
:�
`
training/Adam/m_2Const*
valueB�*    *
dtype0*
_output_shapes	
:�
�
training/Adam/m_2_1VarHandleOp*$
shared_nametraining/Adam/m_2_1*&
_class
loc:@training/Adam/m_2_1*
	container *
shape:�*
dtype0*
_output_shapes
: 
w
4training/Adam/m_2_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/m_2_1*
_output_shapes
: 
c
training/Adam/m_2_1/AssignAssignVariableOptraining/Adam/m_2_1training/Adam/m_2*
dtype0
x
'training/Adam/m_2_1/Read/ReadVariableOpReadVariableOptraining/Adam/m_2_1*
dtype0*
_output_shapes	
:�
`
training/Adam/m_3Const*
valueB�*    *
dtype0*
_output_shapes	
:�
�
training/Adam/m_3_1VarHandleOp*
dtype0*
_output_shapes
: *$
shared_nametraining/Adam/m_3_1*&
_class
loc:@training/Adam/m_3_1*
	container *
shape:�
w
4training/Adam/m_3_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/m_3_1*
_output_shapes
: 
c
training/Adam/m_3_1/AssignAssignVariableOptraining/Adam/m_3_1training/Adam/m_3*
dtype0
x
'training/Adam/m_3_1/Read/ReadVariableOpReadVariableOptraining/Adam/m_3_1*
dtype0*
_output_shapes	
:�
r
!training/Adam/m_4/shape_as_tensorConst*
valueB"�   �   *
dtype0*
_output_shapes
:
\
training/Adam/m_4/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/m_4Fill!training/Adam/m_4/shape_as_tensortraining/Adam/m_4/Const*
T0*

index_type0* 
_output_shapes
:
��
�
training/Adam/m_4_1VarHandleOp*
shape:
��*
dtype0*
_output_shapes
: *$
shared_nametraining/Adam/m_4_1*&
_class
loc:@training/Adam/m_4_1*
	container 
w
4training/Adam/m_4_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/m_4_1*
_output_shapes
: 
c
training/Adam/m_4_1/AssignAssignVariableOptraining/Adam/m_4_1training/Adam/m_4*
dtype0
}
'training/Adam/m_4_1/Read/ReadVariableOpReadVariableOptraining/Adam/m_4_1*
dtype0* 
_output_shapes
:
��
`
training/Adam/m_5Const*
dtype0*
_output_shapes	
:�*
valueB�*    
�
training/Adam/m_5_1VarHandleOp*
dtype0*
_output_shapes
: *$
shared_nametraining/Adam/m_5_1*&
_class
loc:@training/Adam/m_5_1*
	container *
shape:�
w
4training/Adam/m_5_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/m_5_1*
_output_shapes
: 
c
training/Adam/m_5_1/AssignAssignVariableOptraining/Adam/m_5_1training/Adam/m_5*
dtype0
x
'training/Adam/m_5_1/Read/ReadVariableOpReadVariableOptraining/Adam/m_5_1*
dtype0*
_output_shapes	
:�
`
training/Adam/m_6Const*
valueB�*    *
dtype0*
_output_shapes	
:�
�
training/Adam/m_6_1VarHandleOp*
shape:�*
dtype0*
_output_shapes
: *$
shared_nametraining/Adam/m_6_1*&
_class
loc:@training/Adam/m_6_1*
	container 
w
4training/Adam/m_6_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/m_6_1*
_output_shapes
: 
c
training/Adam/m_6_1/AssignAssignVariableOptraining/Adam/m_6_1training/Adam/m_6*
dtype0
x
'training/Adam/m_6_1/Read/ReadVariableOpReadVariableOptraining/Adam/m_6_1*
dtype0*
_output_shapes	
:�
`
training/Adam/m_7Const*
valueB�*    *
dtype0*
_output_shapes	
:�
�
training/Adam/m_7_1VarHandleOp*$
shared_nametraining/Adam/m_7_1*&
_class
loc:@training/Adam/m_7_1*
	container *
shape:�*
dtype0*
_output_shapes
: 
w
4training/Adam/m_7_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/m_7_1*
_output_shapes
: 
c
training/Adam/m_7_1/AssignAssignVariableOptraining/Adam/m_7_1training/Adam/m_7*
dtype0
x
'training/Adam/m_7_1/Read/ReadVariableOpReadVariableOptraining/Adam/m_7_1*
dtype0*
_output_shapes	
:�
r
!training/Adam/m_8/shape_as_tensorConst*
valueB"�   �   *
dtype0*
_output_shapes
:
\
training/Adam/m_8/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
�
training/Adam/m_8Fill!training/Adam/m_8/shape_as_tensortraining/Adam/m_8/Const*
T0*

index_type0* 
_output_shapes
:
��
�
training/Adam/m_8_1VarHandleOp*
dtype0*
_output_shapes
: *$
shared_nametraining/Adam/m_8_1*&
_class
loc:@training/Adam/m_8_1*
	container *
shape:
��
w
4training/Adam/m_8_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/m_8_1*
_output_shapes
: 
c
training/Adam/m_8_1/AssignAssignVariableOptraining/Adam/m_8_1training/Adam/m_8*
dtype0
}
'training/Adam/m_8_1/Read/ReadVariableOpReadVariableOptraining/Adam/m_8_1*
dtype0* 
_output_shapes
:
��
`
training/Adam/m_9Const*
valueB�*    *
dtype0*
_output_shapes	
:�
�
training/Adam/m_9_1VarHandleOp*
	container *
shape:�*
dtype0*
_output_shapes
: *$
shared_nametraining/Adam/m_9_1*&
_class
loc:@training/Adam/m_9_1
w
4training/Adam/m_9_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/m_9_1*
_output_shapes
: 
c
training/Adam/m_9_1/AssignAssignVariableOptraining/Adam/m_9_1training/Adam/m_9*
dtype0
x
'training/Adam/m_9_1/Read/ReadVariableOpReadVariableOptraining/Adam/m_9_1*
dtype0*
_output_shapes	
:�
a
training/Adam/m_10Const*
valueB�*    *
dtype0*
_output_shapes	
:�
�
training/Adam/m_10_1VarHandleOp*
shape:�*
dtype0*
_output_shapes
: *%
shared_nametraining/Adam/m_10_1*'
_class
loc:@training/Adam/m_10_1*
	container 
y
5training/Adam/m_10_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/m_10_1*
_output_shapes
: 
f
training/Adam/m_10_1/AssignAssignVariableOptraining/Adam/m_10_1training/Adam/m_10*
dtype0
z
(training/Adam/m_10_1/Read/ReadVariableOpReadVariableOptraining/Adam/m_10_1*
dtype0*
_output_shapes	
:�
a
training/Adam/m_11Const*
dtype0*
_output_shapes	
:�*
valueB�*    
�
training/Adam/m_11_1VarHandleOp*
dtype0*
_output_shapes
: *%
shared_nametraining/Adam/m_11_1*'
_class
loc:@training/Adam/m_11_1*
	container *
shape:�
y
5training/Adam/m_11_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/m_11_1*
_output_shapes
: 
f
training/Adam/m_11_1/AssignAssignVariableOptraining/Adam/m_11_1training/Adam/m_11*
dtype0
z
(training/Adam/m_11_1/Read/ReadVariableOpReadVariableOptraining/Adam/m_11_1*
dtype0*
_output_shapes	
:�
s
"training/Adam/m_12/shape_as_tensorConst*
valueB"�   �   *
dtype0*
_output_shapes
:
]
training/Adam/m_12/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/m_12Fill"training/Adam/m_12/shape_as_tensortraining/Adam/m_12/Const* 
_output_shapes
:
��*
T0*

index_type0
�
training/Adam/m_12_1VarHandleOp*
dtype0*
_output_shapes
: *%
shared_nametraining/Adam/m_12_1*'
_class
loc:@training/Adam/m_12_1*
	container *
shape:
��
y
5training/Adam/m_12_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/m_12_1*
_output_shapes
: 
f
training/Adam/m_12_1/AssignAssignVariableOptraining/Adam/m_12_1training/Adam/m_12*
dtype0

(training/Adam/m_12_1/Read/ReadVariableOpReadVariableOptraining/Adam/m_12_1*
dtype0* 
_output_shapes
:
��
a
training/Adam/m_13Const*
dtype0*
_output_shapes	
:�*
valueB�*    
�
training/Adam/m_13_1VarHandleOp*%
shared_nametraining/Adam/m_13_1*'
_class
loc:@training/Adam/m_13_1*
	container *
shape:�*
dtype0*
_output_shapes
: 
y
5training/Adam/m_13_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/m_13_1*
_output_shapes
: 
f
training/Adam/m_13_1/AssignAssignVariableOptraining/Adam/m_13_1training/Adam/m_13*
dtype0
z
(training/Adam/m_13_1/Read/ReadVariableOpReadVariableOptraining/Adam/m_13_1*
dtype0*
_output_shapes	
:�
a
training/Adam/m_14Const*
dtype0*
_output_shapes	
:�*
valueB�*    
�
training/Adam/m_14_1VarHandleOp*
shape:�*
dtype0*
_output_shapes
: *%
shared_nametraining/Adam/m_14_1*'
_class
loc:@training/Adam/m_14_1*
	container 
y
5training/Adam/m_14_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/m_14_1*
_output_shapes
: 
f
training/Adam/m_14_1/AssignAssignVariableOptraining/Adam/m_14_1training/Adam/m_14*
dtype0
z
(training/Adam/m_14_1/Read/ReadVariableOpReadVariableOptraining/Adam/m_14_1*
dtype0*
_output_shapes	
:�
a
training/Adam/m_15Const*
valueB�*    *
dtype0*
_output_shapes	
:�
�
training/Adam/m_15_1VarHandleOp*%
shared_nametraining/Adam/m_15_1*'
_class
loc:@training/Adam/m_15_1*
	container *
shape:�*
dtype0*
_output_shapes
: 
y
5training/Adam/m_15_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/m_15_1*
_output_shapes
: 
f
training/Adam/m_15_1/AssignAssignVariableOptraining/Adam/m_15_1training/Adam/m_15*
dtype0
z
(training/Adam/m_15_1/Read/ReadVariableOpReadVariableOptraining/Adam/m_15_1*
dtype0*
_output_shapes	
:�
s
"training/Adam/m_16/shape_as_tensorConst*
valueB"�   +   *
dtype0*
_output_shapes
:
]
training/Adam/m_16/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/m_16Fill"training/Adam/m_16/shape_as_tensortraining/Adam/m_16/Const*
T0*

index_type0*
_output_shapes
:	�+
�
training/Adam/m_16_1VarHandleOp*
dtype0*
_output_shapes
: *%
shared_nametraining/Adam/m_16_1*'
_class
loc:@training/Adam/m_16_1*
	container *
shape:	�+
y
5training/Adam/m_16_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/m_16_1*
_output_shapes
: 
f
training/Adam/m_16_1/AssignAssignVariableOptraining/Adam/m_16_1training/Adam/m_16*
dtype0
~
(training/Adam/m_16_1/Read/ReadVariableOpReadVariableOptraining/Adam/m_16_1*
dtype0*
_output_shapes
:	�+
_
training/Adam/m_17Const*
valueB+*    *
dtype0*
_output_shapes
:+
�
training/Adam/m_17_1VarHandleOp*%
shared_nametraining/Adam/m_17_1*'
_class
loc:@training/Adam/m_17_1*
	container *
shape:+*
dtype0*
_output_shapes
: 
y
5training/Adam/m_17_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/m_17_1*
_output_shapes
: 
f
training/Adam/m_17_1/AssignAssignVariableOptraining/Adam/m_17_1training/Adam/m_17*
dtype0
y
(training/Adam/m_17_1/Read/ReadVariableOpReadVariableOptraining/Adam/m_17_1*
dtype0*
_output_shapes
:+
r
!training/Adam/v_0/shape_as_tensorConst*
valueB"   �   *
dtype0*
_output_shapes
:
\
training/Adam/v_0/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/v_0Fill!training/Adam/v_0/shape_as_tensortraining/Adam/v_0/Const*
T0*

index_type0* 
_output_shapes
:
��
�
training/Adam/v_0_1VarHandleOp*&
_class
loc:@training/Adam/v_0_1*
	container *
shape:
��*
dtype0*
_output_shapes
: *$
shared_nametraining/Adam/v_0_1
w
4training/Adam/v_0_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/v_0_1*
_output_shapes
: 
c
training/Adam/v_0_1/AssignAssignVariableOptraining/Adam/v_0_1training/Adam/v_0*
dtype0
}
'training/Adam/v_0_1/Read/ReadVariableOpReadVariableOptraining/Adam/v_0_1*
dtype0* 
_output_shapes
:
��
`
training/Adam/v_1Const*
valueB�*    *
dtype0*
_output_shapes	
:�
�
training/Adam/v_1_1VarHandleOp*$
shared_nametraining/Adam/v_1_1*&
_class
loc:@training/Adam/v_1_1*
	container *
shape:�*
dtype0*
_output_shapes
: 
w
4training/Adam/v_1_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/v_1_1*
_output_shapes
: 
c
training/Adam/v_1_1/AssignAssignVariableOptraining/Adam/v_1_1training/Adam/v_1*
dtype0
x
'training/Adam/v_1_1/Read/ReadVariableOpReadVariableOptraining/Adam/v_1_1*
dtype0*
_output_shapes	
:�
`
training/Adam/v_2Const*
dtype0*
_output_shapes	
:�*
valueB�*    
�
training/Adam/v_2_1VarHandleOp*
shape:�*
dtype0*
_output_shapes
: *$
shared_nametraining/Adam/v_2_1*&
_class
loc:@training/Adam/v_2_1*
	container 
w
4training/Adam/v_2_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/v_2_1*
_output_shapes
: 
c
training/Adam/v_2_1/AssignAssignVariableOptraining/Adam/v_2_1training/Adam/v_2*
dtype0
x
'training/Adam/v_2_1/Read/ReadVariableOpReadVariableOptraining/Adam/v_2_1*
dtype0*
_output_shapes	
:�
`
training/Adam/v_3Const*
valueB�*    *
dtype0*
_output_shapes	
:�
�
training/Adam/v_3_1VarHandleOp*&
_class
loc:@training/Adam/v_3_1*
	container *
shape:�*
dtype0*
_output_shapes
: *$
shared_nametraining/Adam/v_3_1
w
4training/Adam/v_3_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/v_3_1*
_output_shapes
: 
c
training/Adam/v_3_1/AssignAssignVariableOptraining/Adam/v_3_1training/Adam/v_3*
dtype0
x
'training/Adam/v_3_1/Read/ReadVariableOpReadVariableOptraining/Adam/v_3_1*
dtype0*
_output_shapes	
:�
r
!training/Adam/v_4/shape_as_tensorConst*
valueB"�   �   *
dtype0*
_output_shapes
:
\
training/Adam/v_4/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/v_4Fill!training/Adam/v_4/shape_as_tensortraining/Adam/v_4/Const* 
_output_shapes
:
��*
T0*

index_type0
�
training/Adam/v_4_1VarHandleOp*$
shared_nametraining/Adam/v_4_1*&
_class
loc:@training/Adam/v_4_1*
	container *
shape:
��*
dtype0*
_output_shapes
: 
w
4training/Adam/v_4_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/v_4_1*
_output_shapes
: 
c
training/Adam/v_4_1/AssignAssignVariableOptraining/Adam/v_4_1training/Adam/v_4*
dtype0
}
'training/Adam/v_4_1/Read/ReadVariableOpReadVariableOptraining/Adam/v_4_1*
dtype0* 
_output_shapes
:
��
`
training/Adam/v_5Const*
valueB�*    *
dtype0*
_output_shapes	
:�
�
training/Adam/v_5_1VarHandleOp*
	container *
shape:�*
dtype0*
_output_shapes
: *$
shared_nametraining/Adam/v_5_1*&
_class
loc:@training/Adam/v_5_1
w
4training/Adam/v_5_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/v_5_1*
_output_shapes
: 
c
training/Adam/v_5_1/AssignAssignVariableOptraining/Adam/v_5_1training/Adam/v_5*
dtype0
x
'training/Adam/v_5_1/Read/ReadVariableOpReadVariableOptraining/Adam/v_5_1*
dtype0*
_output_shapes	
:�
`
training/Adam/v_6Const*
valueB�*    *
dtype0*
_output_shapes	
:�
�
training/Adam/v_6_1VarHandleOp*
dtype0*
_output_shapes
: *$
shared_nametraining/Adam/v_6_1*&
_class
loc:@training/Adam/v_6_1*
	container *
shape:�
w
4training/Adam/v_6_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/v_6_1*
_output_shapes
: 
c
training/Adam/v_6_1/AssignAssignVariableOptraining/Adam/v_6_1training/Adam/v_6*
dtype0
x
'training/Adam/v_6_1/Read/ReadVariableOpReadVariableOptraining/Adam/v_6_1*
dtype0*
_output_shapes	
:�
`
training/Adam/v_7Const*
valueB�*    *
dtype0*
_output_shapes	
:�
�
training/Adam/v_7_1VarHandleOp*$
shared_nametraining/Adam/v_7_1*&
_class
loc:@training/Adam/v_7_1*
	container *
shape:�*
dtype0*
_output_shapes
: 
w
4training/Adam/v_7_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/v_7_1*
_output_shapes
: 
c
training/Adam/v_7_1/AssignAssignVariableOptraining/Adam/v_7_1training/Adam/v_7*
dtype0
x
'training/Adam/v_7_1/Read/ReadVariableOpReadVariableOptraining/Adam/v_7_1*
dtype0*
_output_shapes	
:�
r
!training/Adam/v_8/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB"�   �   
\
training/Adam/v_8/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/v_8Fill!training/Adam/v_8/shape_as_tensortraining/Adam/v_8/Const* 
_output_shapes
:
��*
T0*

index_type0
�
training/Adam/v_8_1VarHandleOp*
dtype0*
_output_shapes
: *$
shared_nametraining/Adam/v_8_1*&
_class
loc:@training/Adam/v_8_1*
	container *
shape:
��
w
4training/Adam/v_8_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/v_8_1*
_output_shapes
: 
c
training/Adam/v_8_1/AssignAssignVariableOptraining/Adam/v_8_1training/Adam/v_8*
dtype0
}
'training/Adam/v_8_1/Read/ReadVariableOpReadVariableOptraining/Adam/v_8_1*
dtype0* 
_output_shapes
:
��
`
training/Adam/v_9Const*
valueB�*    *
dtype0*
_output_shapes	
:�
�
training/Adam/v_9_1VarHandleOp*
shape:�*
dtype0*
_output_shapes
: *$
shared_nametraining/Adam/v_9_1*&
_class
loc:@training/Adam/v_9_1*
	container 
w
4training/Adam/v_9_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/v_9_1*
_output_shapes
: 
c
training/Adam/v_9_1/AssignAssignVariableOptraining/Adam/v_9_1training/Adam/v_9*
dtype0
x
'training/Adam/v_9_1/Read/ReadVariableOpReadVariableOptraining/Adam/v_9_1*
dtype0*
_output_shapes	
:�
a
training/Adam/v_10Const*
valueB�*    *
dtype0*
_output_shapes	
:�
�
training/Adam/v_10_1VarHandleOp*%
shared_nametraining/Adam/v_10_1*'
_class
loc:@training/Adam/v_10_1*
	container *
shape:�*
dtype0*
_output_shapes
: 
y
5training/Adam/v_10_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/v_10_1*
_output_shapes
: 
f
training/Adam/v_10_1/AssignAssignVariableOptraining/Adam/v_10_1training/Adam/v_10*
dtype0
z
(training/Adam/v_10_1/Read/ReadVariableOpReadVariableOptraining/Adam/v_10_1*
dtype0*
_output_shapes	
:�
a
training/Adam/v_11Const*
dtype0*
_output_shapes	
:�*
valueB�*    
�
training/Adam/v_11_1VarHandleOp*
dtype0*
_output_shapes
: *%
shared_nametraining/Adam/v_11_1*'
_class
loc:@training/Adam/v_11_1*
	container *
shape:�
y
5training/Adam/v_11_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/v_11_1*
_output_shapes
: 
f
training/Adam/v_11_1/AssignAssignVariableOptraining/Adam/v_11_1training/Adam/v_11*
dtype0
z
(training/Adam/v_11_1/Read/ReadVariableOpReadVariableOptraining/Adam/v_11_1*
dtype0*
_output_shapes	
:�
s
"training/Adam/v_12/shape_as_tensorConst*
valueB"�   �   *
dtype0*
_output_shapes
:
]
training/Adam/v_12/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
�
training/Adam/v_12Fill"training/Adam/v_12/shape_as_tensortraining/Adam/v_12/Const*
T0*

index_type0* 
_output_shapes
:
��
�
training/Adam/v_12_1VarHandleOp*%
shared_nametraining/Adam/v_12_1*'
_class
loc:@training/Adam/v_12_1*
	container *
shape:
��*
dtype0*
_output_shapes
: 
y
5training/Adam/v_12_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/v_12_1*
_output_shapes
: 
f
training/Adam/v_12_1/AssignAssignVariableOptraining/Adam/v_12_1training/Adam/v_12*
dtype0

(training/Adam/v_12_1/Read/ReadVariableOpReadVariableOptraining/Adam/v_12_1*
dtype0* 
_output_shapes
:
��
a
training/Adam/v_13Const*
dtype0*
_output_shapes	
:�*
valueB�*    
�
training/Adam/v_13_1VarHandleOp*%
shared_nametraining/Adam/v_13_1*'
_class
loc:@training/Adam/v_13_1*
	container *
shape:�*
dtype0*
_output_shapes
: 
y
5training/Adam/v_13_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/v_13_1*
_output_shapes
: 
f
training/Adam/v_13_1/AssignAssignVariableOptraining/Adam/v_13_1training/Adam/v_13*
dtype0
z
(training/Adam/v_13_1/Read/ReadVariableOpReadVariableOptraining/Adam/v_13_1*
dtype0*
_output_shapes	
:�
a
training/Adam/v_14Const*
valueB�*    *
dtype0*
_output_shapes	
:�
�
training/Adam/v_14_1VarHandleOp*
dtype0*
_output_shapes
: *%
shared_nametraining/Adam/v_14_1*'
_class
loc:@training/Adam/v_14_1*
	container *
shape:�
y
5training/Adam/v_14_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/v_14_1*
_output_shapes
: 
f
training/Adam/v_14_1/AssignAssignVariableOptraining/Adam/v_14_1training/Adam/v_14*
dtype0
z
(training/Adam/v_14_1/Read/ReadVariableOpReadVariableOptraining/Adam/v_14_1*
dtype0*
_output_shapes	
:�
a
training/Adam/v_15Const*
valueB�*    *
dtype0*
_output_shapes	
:�
�
training/Adam/v_15_1VarHandleOp*
dtype0*
_output_shapes
: *%
shared_nametraining/Adam/v_15_1*'
_class
loc:@training/Adam/v_15_1*
	container *
shape:�
y
5training/Adam/v_15_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/v_15_1*
_output_shapes
: 
f
training/Adam/v_15_1/AssignAssignVariableOptraining/Adam/v_15_1training/Adam/v_15*
dtype0
z
(training/Adam/v_15_1/Read/ReadVariableOpReadVariableOptraining/Adam/v_15_1*
dtype0*
_output_shapes	
:�
s
"training/Adam/v_16/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB"�   +   
]
training/Adam/v_16/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
�
training/Adam/v_16Fill"training/Adam/v_16/shape_as_tensortraining/Adam/v_16/Const*
T0*

index_type0*
_output_shapes
:	�+
�
training/Adam/v_16_1VarHandleOp*
shape:	�+*
dtype0*
_output_shapes
: *%
shared_nametraining/Adam/v_16_1*'
_class
loc:@training/Adam/v_16_1*
	container 
y
5training/Adam/v_16_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/v_16_1*
_output_shapes
: 
f
training/Adam/v_16_1/AssignAssignVariableOptraining/Adam/v_16_1training/Adam/v_16*
dtype0
~
(training/Adam/v_16_1/Read/ReadVariableOpReadVariableOptraining/Adam/v_16_1*
dtype0*
_output_shapes
:	�+
_
training/Adam/v_17Const*
valueB+*    *
dtype0*
_output_shapes
:+
�
training/Adam/v_17_1VarHandleOp*%
shared_nametraining/Adam/v_17_1*'
_class
loc:@training/Adam/v_17_1*
	container *
shape:+*
dtype0*
_output_shapes
: 
y
5training/Adam/v_17_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/v_17_1*
_output_shapes
: 
f
training/Adam/v_17_1/AssignAssignVariableOptraining/Adam/v_17_1training/Adam/v_17*
dtype0
y
(training/Adam/v_17_1/Read/ReadVariableOpReadVariableOptraining/Adam/v_17_1*
dtype0*
_output_shapes
:+
n
$training/Adam/vhat_0/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
_
training/Adam/vhat_0/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/vhat_0Fill$training/Adam/vhat_0/shape_as_tensortraining/Adam/vhat_0/Const*
T0*

index_type0*
_output_shapes
:
�
training/Adam/vhat_0_1VarHandleOp*'
shared_nametraining/Adam/vhat_0_1*)
_class
loc:@training/Adam/vhat_0_1*
	container *
shape:*
dtype0*
_output_shapes
: 
}
7training/Adam/vhat_0_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/vhat_0_1*
_output_shapes
: 
l
training/Adam/vhat_0_1/AssignAssignVariableOptraining/Adam/vhat_0_1training/Adam/vhat_0*
dtype0
}
*training/Adam/vhat_0_1/Read/ReadVariableOpReadVariableOptraining/Adam/vhat_0_1*
dtype0*
_output_shapes
:
n
$training/Adam/vhat_1/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
_
training/Adam/vhat_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/vhat_1Fill$training/Adam/vhat_1/shape_as_tensortraining/Adam/vhat_1/Const*
T0*

index_type0*
_output_shapes
:
�
training/Adam/vhat_1_1VarHandleOp*'
shared_nametraining/Adam/vhat_1_1*)
_class
loc:@training/Adam/vhat_1_1*
	container *
shape:*
dtype0*
_output_shapes
: 
}
7training/Adam/vhat_1_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/vhat_1_1*
_output_shapes
: 
l
training/Adam/vhat_1_1/AssignAssignVariableOptraining/Adam/vhat_1_1training/Adam/vhat_1*
dtype0
}
*training/Adam/vhat_1_1/Read/ReadVariableOpReadVariableOptraining/Adam/vhat_1_1*
dtype0*
_output_shapes
:
n
$training/Adam/vhat_2/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
_
training/Adam/vhat_2/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/vhat_2Fill$training/Adam/vhat_2/shape_as_tensortraining/Adam/vhat_2/Const*
_output_shapes
:*
T0*

index_type0
�
training/Adam/vhat_2_1VarHandleOp*
shape:*
dtype0*
_output_shapes
: *'
shared_nametraining/Adam/vhat_2_1*)
_class
loc:@training/Adam/vhat_2_1*
	container 
}
7training/Adam/vhat_2_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/vhat_2_1*
_output_shapes
: 
l
training/Adam/vhat_2_1/AssignAssignVariableOptraining/Adam/vhat_2_1training/Adam/vhat_2*
dtype0
}
*training/Adam/vhat_2_1/Read/ReadVariableOpReadVariableOptraining/Adam/vhat_2_1*
dtype0*
_output_shapes
:
n
$training/Adam/vhat_3/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
_
training/Adam/vhat_3/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/vhat_3Fill$training/Adam/vhat_3/shape_as_tensortraining/Adam/vhat_3/Const*
T0*

index_type0*
_output_shapes
:
�
training/Adam/vhat_3_1VarHandleOp*'
shared_nametraining/Adam/vhat_3_1*)
_class
loc:@training/Adam/vhat_3_1*
	container *
shape:*
dtype0*
_output_shapes
: 
}
7training/Adam/vhat_3_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/vhat_3_1*
_output_shapes
: 
l
training/Adam/vhat_3_1/AssignAssignVariableOptraining/Adam/vhat_3_1training/Adam/vhat_3*
dtype0
}
*training/Adam/vhat_3_1/Read/ReadVariableOpReadVariableOptraining/Adam/vhat_3_1*
dtype0*
_output_shapes
:
n
$training/Adam/vhat_4/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
_
training/Adam/vhat_4/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/vhat_4Fill$training/Adam/vhat_4/shape_as_tensortraining/Adam/vhat_4/Const*
T0*

index_type0*
_output_shapes
:
�
training/Adam/vhat_4_1VarHandleOp*'
shared_nametraining/Adam/vhat_4_1*)
_class
loc:@training/Adam/vhat_4_1*
	container *
shape:*
dtype0*
_output_shapes
: 
}
7training/Adam/vhat_4_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/vhat_4_1*
_output_shapes
: 
l
training/Adam/vhat_4_1/AssignAssignVariableOptraining/Adam/vhat_4_1training/Adam/vhat_4*
dtype0
}
*training/Adam/vhat_4_1/Read/ReadVariableOpReadVariableOptraining/Adam/vhat_4_1*
dtype0*
_output_shapes
:
n
$training/Adam/vhat_5/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
_
training/Adam/vhat_5/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/vhat_5Fill$training/Adam/vhat_5/shape_as_tensortraining/Adam/vhat_5/Const*
T0*

index_type0*
_output_shapes
:
�
training/Adam/vhat_5_1VarHandleOp*'
shared_nametraining/Adam/vhat_5_1*)
_class
loc:@training/Adam/vhat_5_1*
	container *
shape:*
dtype0*
_output_shapes
: 
}
7training/Adam/vhat_5_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/vhat_5_1*
_output_shapes
: 
l
training/Adam/vhat_5_1/AssignAssignVariableOptraining/Adam/vhat_5_1training/Adam/vhat_5*
dtype0
}
*training/Adam/vhat_5_1/Read/ReadVariableOpReadVariableOptraining/Adam/vhat_5_1*
dtype0*
_output_shapes
:
n
$training/Adam/vhat_6/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
_
training/Adam/vhat_6/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
�
training/Adam/vhat_6Fill$training/Adam/vhat_6/shape_as_tensortraining/Adam/vhat_6/Const*
T0*

index_type0*
_output_shapes
:
�
training/Adam/vhat_6_1VarHandleOp*'
shared_nametraining/Adam/vhat_6_1*)
_class
loc:@training/Adam/vhat_6_1*
	container *
shape:*
dtype0*
_output_shapes
: 
}
7training/Adam/vhat_6_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/vhat_6_1*
_output_shapes
: 
l
training/Adam/vhat_6_1/AssignAssignVariableOptraining/Adam/vhat_6_1training/Adam/vhat_6*
dtype0
}
*training/Adam/vhat_6_1/Read/ReadVariableOpReadVariableOptraining/Adam/vhat_6_1*
dtype0*
_output_shapes
:
n
$training/Adam/vhat_7/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
_
training/Adam/vhat_7/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/vhat_7Fill$training/Adam/vhat_7/shape_as_tensortraining/Adam/vhat_7/Const*
T0*

index_type0*
_output_shapes
:
�
training/Adam/vhat_7_1VarHandleOp*
dtype0*
_output_shapes
: *'
shared_nametraining/Adam/vhat_7_1*)
_class
loc:@training/Adam/vhat_7_1*
	container *
shape:
}
7training/Adam/vhat_7_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/vhat_7_1*
_output_shapes
: 
l
training/Adam/vhat_7_1/AssignAssignVariableOptraining/Adam/vhat_7_1training/Adam/vhat_7*
dtype0
}
*training/Adam/vhat_7_1/Read/ReadVariableOpReadVariableOptraining/Adam/vhat_7_1*
dtype0*
_output_shapes
:
n
$training/Adam/vhat_8/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
_
training/Adam/vhat_8/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/vhat_8Fill$training/Adam/vhat_8/shape_as_tensortraining/Adam/vhat_8/Const*
T0*

index_type0*
_output_shapes
:
�
training/Adam/vhat_8_1VarHandleOp*
dtype0*
_output_shapes
: *'
shared_nametraining/Adam/vhat_8_1*)
_class
loc:@training/Adam/vhat_8_1*
	container *
shape:
}
7training/Adam/vhat_8_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/vhat_8_1*
_output_shapes
: 
l
training/Adam/vhat_8_1/AssignAssignVariableOptraining/Adam/vhat_8_1training/Adam/vhat_8*
dtype0
}
*training/Adam/vhat_8_1/Read/ReadVariableOpReadVariableOptraining/Adam/vhat_8_1*
dtype0*
_output_shapes
:
n
$training/Adam/vhat_9/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
_
training/Adam/vhat_9/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/vhat_9Fill$training/Adam/vhat_9/shape_as_tensortraining/Adam/vhat_9/Const*
T0*

index_type0*
_output_shapes
:
�
training/Adam/vhat_9_1VarHandleOp*
dtype0*
_output_shapes
: *'
shared_nametraining/Adam/vhat_9_1*)
_class
loc:@training/Adam/vhat_9_1*
	container *
shape:
}
7training/Adam/vhat_9_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/vhat_9_1*
_output_shapes
: 
l
training/Adam/vhat_9_1/AssignAssignVariableOptraining/Adam/vhat_9_1training/Adam/vhat_9*
dtype0
}
*training/Adam/vhat_9_1/Read/ReadVariableOpReadVariableOptraining/Adam/vhat_9_1*
dtype0*
_output_shapes
:
o
%training/Adam/vhat_10/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
`
training/Adam/vhat_10/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/vhat_10Fill%training/Adam/vhat_10/shape_as_tensortraining/Adam/vhat_10/Const*
_output_shapes
:*
T0*

index_type0
�
training/Adam/vhat_10_1VarHandleOp*(
shared_nametraining/Adam/vhat_10_1**
_class 
loc:@training/Adam/vhat_10_1*
	container *
shape:*
dtype0*
_output_shapes
: 

8training/Adam/vhat_10_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/vhat_10_1*
_output_shapes
: 
o
training/Adam/vhat_10_1/AssignAssignVariableOptraining/Adam/vhat_10_1training/Adam/vhat_10*
dtype0

+training/Adam/vhat_10_1/Read/ReadVariableOpReadVariableOptraining/Adam/vhat_10_1*
dtype0*
_output_shapes
:
o
%training/Adam/vhat_11/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
`
training/Adam/vhat_11/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/vhat_11Fill%training/Adam/vhat_11/shape_as_tensortraining/Adam/vhat_11/Const*
T0*

index_type0*
_output_shapes
:
�
training/Adam/vhat_11_1VarHandleOp*(
shared_nametraining/Adam/vhat_11_1**
_class 
loc:@training/Adam/vhat_11_1*
	container *
shape:*
dtype0*
_output_shapes
: 

8training/Adam/vhat_11_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/vhat_11_1*
_output_shapes
: 
o
training/Adam/vhat_11_1/AssignAssignVariableOptraining/Adam/vhat_11_1training/Adam/vhat_11*
dtype0

+training/Adam/vhat_11_1/Read/ReadVariableOpReadVariableOptraining/Adam/vhat_11_1*
dtype0*
_output_shapes
:
o
%training/Adam/vhat_12/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
`
training/Adam/vhat_12/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/vhat_12Fill%training/Adam/vhat_12/shape_as_tensortraining/Adam/vhat_12/Const*
T0*

index_type0*
_output_shapes
:
�
training/Adam/vhat_12_1VarHandleOp*
dtype0*
_output_shapes
: *(
shared_nametraining/Adam/vhat_12_1**
_class 
loc:@training/Adam/vhat_12_1*
	container *
shape:

8training/Adam/vhat_12_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/vhat_12_1*
_output_shapes
: 
o
training/Adam/vhat_12_1/AssignAssignVariableOptraining/Adam/vhat_12_1training/Adam/vhat_12*
dtype0

+training/Adam/vhat_12_1/Read/ReadVariableOpReadVariableOptraining/Adam/vhat_12_1*
dtype0*
_output_shapes
:
o
%training/Adam/vhat_13/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
`
training/Adam/vhat_13/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/vhat_13Fill%training/Adam/vhat_13/shape_as_tensortraining/Adam/vhat_13/Const*
T0*

index_type0*
_output_shapes
:
�
training/Adam/vhat_13_1VarHandleOp**
_class 
loc:@training/Adam/vhat_13_1*
	container *
shape:*
dtype0*
_output_shapes
: *(
shared_nametraining/Adam/vhat_13_1

8training/Adam/vhat_13_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/vhat_13_1*
_output_shapes
: 
o
training/Adam/vhat_13_1/AssignAssignVariableOptraining/Adam/vhat_13_1training/Adam/vhat_13*
dtype0

+training/Adam/vhat_13_1/Read/ReadVariableOpReadVariableOptraining/Adam/vhat_13_1*
dtype0*
_output_shapes
:
o
%training/Adam/vhat_14/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
`
training/Adam/vhat_14/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/vhat_14Fill%training/Adam/vhat_14/shape_as_tensortraining/Adam/vhat_14/Const*
T0*

index_type0*
_output_shapes
:
�
training/Adam/vhat_14_1VarHandleOp*
	container *
shape:*
dtype0*
_output_shapes
: *(
shared_nametraining/Adam/vhat_14_1**
_class 
loc:@training/Adam/vhat_14_1

8training/Adam/vhat_14_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/vhat_14_1*
_output_shapes
: 
o
training/Adam/vhat_14_1/AssignAssignVariableOptraining/Adam/vhat_14_1training/Adam/vhat_14*
dtype0

+training/Adam/vhat_14_1/Read/ReadVariableOpReadVariableOptraining/Adam/vhat_14_1*
dtype0*
_output_shapes
:
o
%training/Adam/vhat_15/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
`
training/Adam/vhat_15/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/vhat_15Fill%training/Adam/vhat_15/shape_as_tensortraining/Adam/vhat_15/Const*
T0*

index_type0*
_output_shapes
:
�
training/Adam/vhat_15_1VarHandleOp*(
shared_nametraining/Adam/vhat_15_1**
_class 
loc:@training/Adam/vhat_15_1*
	container *
shape:*
dtype0*
_output_shapes
: 

8training/Adam/vhat_15_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/vhat_15_1*
_output_shapes
: 
o
training/Adam/vhat_15_1/AssignAssignVariableOptraining/Adam/vhat_15_1training/Adam/vhat_15*
dtype0

+training/Adam/vhat_15_1/Read/ReadVariableOpReadVariableOptraining/Adam/vhat_15_1*
dtype0*
_output_shapes
:
o
%training/Adam/vhat_16/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
`
training/Adam/vhat_16/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/vhat_16Fill%training/Adam/vhat_16/shape_as_tensortraining/Adam/vhat_16/Const*
T0*

index_type0*
_output_shapes
:
�
training/Adam/vhat_16_1VarHandleOp*
	container *
shape:*
dtype0*
_output_shapes
: *(
shared_nametraining/Adam/vhat_16_1**
_class 
loc:@training/Adam/vhat_16_1

8training/Adam/vhat_16_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/vhat_16_1*
_output_shapes
: 
o
training/Adam/vhat_16_1/AssignAssignVariableOptraining/Adam/vhat_16_1training/Adam/vhat_16*
dtype0

+training/Adam/vhat_16_1/Read/ReadVariableOpReadVariableOptraining/Adam/vhat_16_1*
dtype0*
_output_shapes
:
o
%training/Adam/vhat_17/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
`
training/Adam/vhat_17/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
�
training/Adam/vhat_17Fill%training/Adam/vhat_17/shape_as_tensortraining/Adam/vhat_17/Const*
T0*

index_type0*
_output_shapes
:
�
training/Adam/vhat_17_1VarHandleOp*(
shared_nametraining/Adam/vhat_17_1**
_class 
loc:@training/Adam/vhat_17_1*
	container *
shape:*
dtype0*
_output_shapes
: 

8training/Adam/vhat_17_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/vhat_17_1*
_output_shapes
: 
o
training/Adam/vhat_17_1/AssignAssignVariableOptraining/Adam/vhat_17_1training/Adam/vhat_17*
dtype0

+training/Adam/vhat_17_1/Read/ReadVariableOpReadVariableOptraining/Adam/vhat_17_1*
dtype0*
_output_shapes
:
b
training/Adam/ReadVariableOp_2ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
x
"training/Adam/mul_1/ReadVariableOpReadVariableOptraining/Adam/m_0_1*
dtype0* 
_output_shapes
:
��
�
training/Adam/mul_1Multraining/Adam/ReadVariableOp_2"training/Adam/mul_1/ReadVariableOp*
T0* 
_output_shapes
:
��
b
training/Adam/ReadVariableOp_3ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
Z
training/Adam/sub_2/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
r
training/Adam/sub_2Subtraining/Adam/sub_2/xtraining/Adam/ReadVariableOp_3*
T0*
_output_shapes
: 
�
training/Adam/mul_2Multraining/Adam/sub_24training/Adam/gradients/dense_1/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:
��
q
training/Adam/add_1AddV2training/Adam/mul_1training/Adam/mul_2* 
_output_shapes
:
��*
T0
b
training/Adam/ReadVariableOp_4ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
x
"training/Adam/mul_3/ReadVariableOpReadVariableOptraining/Adam/v_0_1*
dtype0* 
_output_shapes
:
��
�
training/Adam/mul_3Multraining/Adam/ReadVariableOp_4"training/Adam/mul_3/ReadVariableOp* 
_output_shapes
:
��*
T0
b
training/Adam/ReadVariableOp_5ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
Z
training/Adam/sub_3/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
r
training/Adam/sub_3Subtraining/Adam/sub_3/xtraining/Adam/ReadVariableOp_5*
_output_shapes
: *
T0

training/Adam/SquareSquare4training/Adam/gradients/dense_1/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:
��
p
training/Adam/mul_4Multraining/Adam/sub_3training/Adam/Square*
T0* 
_output_shapes
:
��
q
training/Adam/add_2AddV2training/Adam/mul_3training/Adam/mul_4* 
_output_shapes
:
��*
T0
m
training/Adam/mul_5Multraining/Adam/multraining/Adam/add_1*
T0* 
_output_shapes
:
��
Z
training/Adam/Const_3Const*
dtype0*
_output_shapes
: *
valueB
 *    
Z
training/Adam/Const_4Const*
valueB
 *  �*
dtype0*
_output_shapes
: 
�
%training/Adam/clip_by_value_1/MinimumMinimumtraining/Adam/add_2training/Adam/Const_4* 
_output_shapes
:
��*
T0
�
training/Adam/clip_by_value_1Maximum%training/Adam/clip_by_value_1/Minimumtraining/Adam/Const_3*
T0* 
_output_shapes
:
��
f
training/Adam/Sqrt_1Sqrttraining/Adam/clip_by_value_1*
T0* 
_output_shapes
:
��
Z
training/Adam/add_3/yConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
t
training/Adam/add_3AddV2training/Adam/Sqrt_1training/Adam/add_3/y*
T0* 
_output_shapes
:
��
w
training/Adam/truediv_1RealDivtraining/Adam/mul_5training/Adam/add_3* 
_output_shapes
:
��*
T0
o
training/Adam/ReadVariableOp_6ReadVariableOpdense_1/kernel*
dtype0* 
_output_shapes
:
��
~
training/Adam/sub_4Subtraining/Adam/ReadVariableOp_6training/Adam/truediv_1* 
_output_shapes
:
��*
T0
i
training/Adam/AssignVariableOpAssignVariableOptraining/Adam/m_0_1training/Adam/add_1*
dtype0
�
training/Adam/ReadVariableOp_7ReadVariableOptraining/Adam/m_0_1^training/Adam/AssignVariableOp*
dtype0* 
_output_shapes
:
��
k
 training/Adam/AssignVariableOp_1AssignVariableOptraining/Adam/v_0_1training/Adam/add_2*
dtype0
�
training/Adam/ReadVariableOp_8ReadVariableOptraining/Adam/v_0_1!^training/Adam/AssignVariableOp_1*
dtype0* 
_output_shapes
:
��
f
 training/Adam/AssignVariableOp_2AssignVariableOpdense_1/kerneltraining/Adam/sub_4*
dtype0
�
training/Adam/ReadVariableOp_9ReadVariableOpdense_1/kernel!^training/Adam/AssignVariableOp_2*
dtype0* 
_output_shapes
:
��
c
training/Adam/ReadVariableOp_10ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
s
"training/Adam/mul_6/ReadVariableOpReadVariableOptraining/Adam/m_1_1*
dtype0*
_output_shapes	
:�
�
training/Adam/mul_6Multraining/Adam/ReadVariableOp_10"training/Adam/mul_6/ReadVariableOp*
T0*
_output_shapes	
:�
c
training/Adam/ReadVariableOp_11ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
Z
training/Adam/sub_5/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
s
training/Adam/sub_5Subtraining/Adam/sub_5/xtraining/Adam/ReadVariableOp_11*
T0*
_output_shapes
: 
�
training/Adam/mul_7Multraining/Adam/sub_58training/Adam/gradients/dense_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:�
l
training/Adam/add_4AddV2training/Adam/mul_6training/Adam/mul_7*
_output_shapes	
:�*
T0
c
training/Adam/ReadVariableOp_12ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
s
"training/Adam/mul_8/ReadVariableOpReadVariableOptraining/Adam/v_1_1*
dtype0*
_output_shapes	
:�
�
training/Adam/mul_8Multraining/Adam/ReadVariableOp_12"training/Adam/mul_8/ReadVariableOp*
_output_shapes	
:�*
T0
c
training/Adam/ReadVariableOp_13ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
Z
training/Adam/sub_6/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
s
training/Adam/sub_6Subtraining/Adam/sub_6/xtraining/Adam/ReadVariableOp_13*
T0*
_output_shapes
: 
�
training/Adam/Square_1Square8training/Adam/gradients/dense_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:�
m
training/Adam/mul_9Multraining/Adam/sub_6training/Adam/Square_1*
T0*
_output_shapes	
:�
l
training/Adam/add_5AddV2training/Adam/mul_8training/Adam/mul_9*
_output_shapes	
:�*
T0
i
training/Adam/mul_10Multraining/Adam/multraining/Adam/add_4*
T0*
_output_shapes	
:�
Z
training/Adam/Const_5Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_6Const*
dtype0*
_output_shapes
: *
valueB
 *  �
�
%training/Adam/clip_by_value_2/MinimumMinimumtraining/Adam/add_5training/Adam/Const_6*
T0*
_output_shapes	
:�
�
training/Adam/clip_by_value_2Maximum%training/Adam/clip_by_value_2/Minimumtraining/Adam/Const_5*
T0*
_output_shapes	
:�
a
training/Adam/Sqrt_2Sqrttraining/Adam/clip_by_value_2*
_output_shapes	
:�*
T0
Z
training/Adam/add_6/yConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
o
training/Adam/add_6AddV2training/Adam/Sqrt_2training/Adam/add_6/y*
_output_shapes	
:�*
T0
s
training/Adam/truediv_2RealDivtraining/Adam/mul_10training/Adam/add_6*
T0*
_output_shapes	
:�
i
training/Adam/ReadVariableOp_14ReadVariableOpdense_1/bias*
dtype0*
_output_shapes	
:�
z
training/Adam/sub_7Subtraining/Adam/ReadVariableOp_14training/Adam/truediv_2*
T0*
_output_shapes	
:�
k
 training/Adam/AssignVariableOp_3AssignVariableOptraining/Adam/m_1_1training/Adam/add_4*
dtype0
�
training/Adam/ReadVariableOp_15ReadVariableOptraining/Adam/m_1_1!^training/Adam/AssignVariableOp_3*
dtype0*
_output_shapes	
:�
k
 training/Adam/AssignVariableOp_4AssignVariableOptraining/Adam/v_1_1training/Adam/add_5*
dtype0
�
training/Adam/ReadVariableOp_16ReadVariableOptraining/Adam/v_1_1!^training/Adam/AssignVariableOp_4*
dtype0*
_output_shapes	
:�
d
 training/Adam/AssignVariableOp_5AssignVariableOpdense_1/biastraining/Adam/sub_7*
dtype0
�
training/Adam/ReadVariableOp_17ReadVariableOpdense_1/bias!^training/Adam/AssignVariableOp_5*
dtype0*
_output_shapes	
:�
c
training/Adam/ReadVariableOp_18ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
t
#training/Adam/mul_11/ReadVariableOpReadVariableOptraining/Adam/m_2_1*
dtype0*
_output_shapes	
:�
�
training/Adam/mul_11Multraining/Adam/ReadVariableOp_18#training/Adam/mul_11/ReadVariableOp*
_output_shapes	
:�*
T0
c
training/Adam/ReadVariableOp_19ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
Z
training/Adam/sub_8/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
s
training/Adam/sub_8Subtraining/Adam/sub_8/xtraining/Adam/ReadVariableOp_19*
T0*
_output_shapes
: 
w
training/Adam/mul_12Multraining/Adam/sub_8training/Adam/gradients/AddN_20*
T0*
_output_shapes	
:�
n
training/Adam/add_7AddV2training/Adam/mul_11training/Adam/mul_12*
T0*
_output_shapes	
:�
c
training/Adam/ReadVariableOp_20ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
t
#training/Adam/mul_13/ReadVariableOpReadVariableOptraining/Adam/v_2_1*
dtype0*
_output_shapes	
:�
�
training/Adam/mul_13Multraining/Adam/ReadVariableOp_20#training/Adam/mul_13/ReadVariableOp*
T0*
_output_shapes	
:�
c
training/Adam/ReadVariableOp_21ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
Z
training/Adam/sub_9/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
s
training/Adam/sub_9Subtraining/Adam/sub_9/xtraining/Adam/ReadVariableOp_21*
T0*
_output_shapes
: 
g
training/Adam/Square_2Squaretraining/Adam/gradients/AddN_20*
_output_shapes	
:�*
T0
n
training/Adam/mul_14Multraining/Adam/sub_9training/Adam/Square_2*
T0*
_output_shapes	
:�
n
training/Adam/add_8AddV2training/Adam/mul_13training/Adam/mul_14*
T0*
_output_shapes	
:�
i
training/Adam/mul_15Multraining/Adam/multraining/Adam/add_7*
T0*
_output_shapes	
:�
Z
training/Adam/Const_7Const*
dtype0*
_output_shapes
: *
valueB
 *    
Z
training/Adam/Const_8Const*
valueB
 *  �*
dtype0*
_output_shapes
: 
�
%training/Adam/clip_by_value_3/MinimumMinimumtraining/Adam/add_8training/Adam/Const_8*
T0*
_output_shapes	
:�
�
training/Adam/clip_by_value_3Maximum%training/Adam/clip_by_value_3/Minimumtraining/Adam/Const_7*
T0*
_output_shapes	
:�
a
training/Adam/Sqrt_3Sqrttraining/Adam/clip_by_value_3*
_output_shapes	
:�*
T0
Z
training/Adam/add_9/yConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
o
training/Adam/add_9AddV2training/Adam/Sqrt_3training/Adam/add_9/y*
_output_shapes	
:�*
T0
s
training/Adam/truediv_3RealDivtraining/Adam/mul_15training/Adam/add_9*
T0*
_output_shapes	
:�
x
training/Adam/ReadVariableOp_22ReadVariableOpbatch_normalization_1/gamma*
dtype0*
_output_shapes	
:�
{
training/Adam/sub_10Subtraining/Adam/ReadVariableOp_22training/Adam/truediv_3*
_output_shapes	
:�*
T0
k
 training/Adam/AssignVariableOp_6AssignVariableOptraining/Adam/m_2_1training/Adam/add_7*
dtype0
�
training/Adam/ReadVariableOp_23ReadVariableOptraining/Adam/m_2_1!^training/Adam/AssignVariableOp_6*
dtype0*
_output_shapes	
:�
k
 training/Adam/AssignVariableOp_7AssignVariableOptraining/Adam/v_2_1training/Adam/add_8*
dtype0
�
training/Adam/ReadVariableOp_24ReadVariableOptraining/Adam/v_2_1!^training/Adam/AssignVariableOp_7*
dtype0*
_output_shapes	
:�
t
 training/Adam/AssignVariableOp_8AssignVariableOpbatch_normalization_1/gammatraining/Adam/sub_10*
dtype0
�
training/Adam/ReadVariableOp_25ReadVariableOpbatch_normalization_1/gamma!^training/Adam/AssignVariableOp_8*
dtype0*
_output_shapes	
:�
c
training/Adam/ReadVariableOp_26ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
t
#training/Adam/mul_16/ReadVariableOpReadVariableOptraining/Adam/m_3_1*
dtype0*
_output_shapes	
:�
�
training/Adam/mul_16Multraining/Adam/ReadVariableOp_26#training/Adam/mul_16/ReadVariableOp*
_output_shapes	
:�*
T0
c
training/Adam/ReadVariableOp_27ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
[
training/Adam/sub_11/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
u
training/Adam/sub_11Subtraining/Adam/sub_11/xtraining/Adam/ReadVariableOp_27*
T0*
_output_shapes
: 
x
training/Adam/mul_17Multraining/Adam/sub_11training/Adam/gradients/AddN_18*
T0*
_output_shapes	
:�
o
training/Adam/add_10AddV2training/Adam/mul_16training/Adam/mul_17*
T0*
_output_shapes	
:�
c
training/Adam/ReadVariableOp_28ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
t
#training/Adam/mul_18/ReadVariableOpReadVariableOptraining/Adam/v_3_1*
dtype0*
_output_shapes	
:�
�
training/Adam/mul_18Multraining/Adam/ReadVariableOp_28#training/Adam/mul_18/ReadVariableOp*
T0*
_output_shapes	
:�
c
training/Adam/ReadVariableOp_29ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
[
training/Adam/sub_12/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
u
training/Adam/sub_12Subtraining/Adam/sub_12/xtraining/Adam/ReadVariableOp_29*
T0*
_output_shapes
: 
g
training/Adam/Square_3Squaretraining/Adam/gradients/AddN_18*
T0*
_output_shapes	
:�
o
training/Adam/mul_19Multraining/Adam/sub_12training/Adam/Square_3*
T0*
_output_shapes	
:�
o
training/Adam/add_11AddV2training/Adam/mul_18training/Adam/mul_19*
_output_shapes	
:�*
T0
j
training/Adam/mul_20Multraining/Adam/multraining/Adam/add_10*
T0*
_output_shapes	
:�
Z
training/Adam/Const_9Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_10Const*
valueB
 *  �*
dtype0*
_output_shapes
: 
�
%training/Adam/clip_by_value_4/MinimumMinimumtraining/Adam/add_11training/Adam/Const_10*
_output_shapes	
:�*
T0
�
training/Adam/clip_by_value_4Maximum%training/Adam/clip_by_value_4/Minimumtraining/Adam/Const_9*
T0*
_output_shapes	
:�
a
training/Adam/Sqrt_4Sqrttraining/Adam/clip_by_value_4*
T0*
_output_shapes	
:�
[
training/Adam/add_12/yConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
q
training/Adam/add_12AddV2training/Adam/Sqrt_4training/Adam/add_12/y*
T0*
_output_shapes	
:�
t
training/Adam/truediv_4RealDivtraining/Adam/mul_20training/Adam/add_12*
_output_shapes	
:�*
T0
w
training/Adam/ReadVariableOp_30ReadVariableOpbatch_normalization_1/beta*
dtype0*
_output_shapes	
:�
{
training/Adam/sub_13Subtraining/Adam/ReadVariableOp_30training/Adam/truediv_4*
T0*
_output_shapes	
:�
l
 training/Adam/AssignVariableOp_9AssignVariableOptraining/Adam/m_3_1training/Adam/add_10*
dtype0
�
training/Adam/ReadVariableOp_31ReadVariableOptraining/Adam/m_3_1!^training/Adam/AssignVariableOp_9*
dtype0*
_output_shapes	
:�
m
!training/Adam/AssignVariableOp_10AssignVariableOptraining/Adam/v_3_1training/Adam/add_11*
dtype0
�
training/Adam/ReadVariableOp_32ReadVariableOptraining/Adam/v_3_1"^training/Adam/AssignVariableOp_10*
dtype0*
_output_shapes	
:�
t
!training/Adam/AssignVariableOp_11AssignVariableOpbatch_normalization_1/betatraining/Adam/sub_13*
dtype0
�
training/Adam/ReadVariableOp_33ReadVariableOpbatch_normalization_1/beta"^training/Adam/AssignVariableOp_11*
dtype0*
_output_shapes	
:�
c
training/Adam/ReadVariableOp_34ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
y
#training/Adam/mul_21/ReadVariableOpReadVariableOptraining/Adam/m_4_1*
dtype0* 
_output_shapes
:
��
�
training/Adam/mul_21Multraining/Adam/ReadVariableOp_34#training/Adam/mul_21/ReadVariableOp*
T0* 
_output_shapes
:
��
c
training/Adam/ReadVariableOp_35ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
[
training/Adam/sub_14/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
u
training/Adam/sub_14Subtraining/Adam/sub_14/xtraining/Adam/ReadVariableOp_35*
T0*
_output_shapes
: 
�
training/Adam/mul_22Multraining/Adam/sub_144training/Adam/gradients/dense_2/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:
��
t
training/Adam/add_13AddV2training/Adam/mul_21training/Adam/mul_22*
T0* 
_output_shapes
:
��
c
training/Adam/ReadVariableOp_36ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
y
#training/Adam/mul_23/ReadVariableOpReadVariableOptraining/Adam/v_4_1*
dtype0* 
_output_shapes
:
��
�
training/Adam/mul_23Multraining/Adam/ReadVariableOp_36#training/Adam/mul_23/ReadVariableOp*
T0* 
_output_shapes
:
��
c
training/Adam/ReadVariableOp_37ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
[
training/Adam/sub_15/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
u
training/Adam/sub_15Subtraining/Adam/sub_15/xtraining/Adam/ReadVariableOp_37*
_output_shapes
: *
T0
�
training/Adam/Square_4Square4training/Adam/gradients/dense_2/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:
��
t
training/Adam/mul_24Multraining/Adam/sub_15training/Adam/Square_4* 
_output_shapes
:
��*
T0
t
training/Adam/add_14AddV2training/Adam/mul_23training/Adam/mul_24*
T0* 
_output_shapes
:
��
o
training/Adam/mul_25Multraining/Adam/multraining/Adam/add_13*
T0* 
_output_shapes
:
��
[
training/Adam/Const_11Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_12Const*
dtype0*
_output_shapes
: *
valueB
 *  �
�
%training/Adam/clip_by_value_5/MinimumMinimumtraining/Adam/add_14training/Adam/Const_12*
T0* 
_output_shapes
:
��
�
training/Adam/clip_by_value_5Maximum%training/Adam/clip_by_value_5/Minimumtraining/Adam/Const_11* 
_output_shapes
:
��*
T0
f
training/Adam/Sqrt_5Sqrttraining/Adam/clip_by_value_5*
T0* 
_output_shapes
:
��
[
training/Adam/add_15/yConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
v
training/Adam/add_15AddV2training/Adam/Sqrt_5training/Adam/add_15/y*
T0* 
_output_shapes
:
��
y
training/Adam/truediv_5RealDivtraining/Adam/mul_25training/Adam/add_15*
T0* 
_output_shapes
:
��
p
training/Adam/ReadVariableOp_38ReadVariableOpdense_2/kernel*
dtype0* 
_output_shapes
:
��
�
training/Adam/sub_16Subtraining/Adam/ReadVariableOp_38training/Adam/truediv_5*
T0* 
_output_shapes
:
��
m
!training/Adam/AssignVariableOp_12AssignVariableOptraining/Adam/m_4_1training/Adam/add_13*
dtype0
�
training/Adam/ReadVariableOp_39ReadVariableOptraining/Adam/m_4_1"^training/Adam/AssignVariableOp_12*
dtype0* 
_output_shapes
:
��
m
!training/Adam/AssignVariableOp_13AssignVariableOptraining/Adam/v_4_1training/Adam/add_14*
dtype0
�
training/Adam/ReadVariableOp_40ReadVariableOptraining/Adam/v_4_1"^training/Adam/AssignVariableOp_13*
dtype0* 
_output_shapes
:
��
h
!training/Adam/AssignVariableOp_14AssignVariableOpdense_2/kerneltraining/Adam/sub_16*
dtype0
�
training/Adam/ReadVariableOp_41ReadVariableOpdense_2/kernel"^training/Adam/AssignVariableOp_14*
dtype0* 
_output_shapes
:
��
c
training/Adam/ReadVariableOp_42ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
t
#training/Adam/mul_26/ReadVariableOpReadVariableOptraining/Adam/m_5_1*
dtype0*
_output_shapes	
:�
�
training/Adam/mul_26Multraining/Adam/ReadVariableOp_42#training/Adam/mul_26/ReadVariableOp*
T0*
_output_shapes	
:�
c
training/Adam/ReadVariableOp_43ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
[
training/Adam/sub_17/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
u
training/Adam/sub_17Subtraining/Adam/sub_17/xtraining/Adam/ReadVariableOp_43*
T0*
_output_shapes
: 
�
training/Adam/mul_27Multraining/Adam/sub_178training/Adam/gradients/dense_2/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:�
o
training/Adam/add_16AddV2training/Adam/mul_26training/Adam/mul_27*
T0*
_output_shapes	
:�
c
training/Adam/ReadVariableOp_44ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
t
#training/Adam/mul_28/ReadVariableOpReadVariableOptraining/Adam/v_5_1*
dtype0*
_output_shapes	
:�
�
training/Adam/mul_28Multraining/Adam/ReadVariableOp_44#training/Adam/mul_28/ReadVariableOp*
T0*
_output_shapes	
:�
c
training/Adam/ReadVariableOp_45ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
[
training/Adam/sub_18/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
u
training/Adam/sub_18Subtraining/Adam/sub_18/xtraining/Adam/ReadVariableOp_45*
T0*
_output_shapes
: 
�
training/Adam/Square_5Square8training/Adam/gradients/dense_2/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:�
o
training/Adam/mul_29Multraining/Adam/sub_18training/Adam/Square_5*
T0*
_output_shapes	
:�
o
training/Adam/add_17AddV2training/Adam/mul_28training/Adam/mul_29*
T0*
_output_shapes	
:�
j
training/Adam/mul_30Multraining/Adam/multraining/Adam/add_16*
T0*
_output_shapes	
:�
[
training/Adam/Const_13Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_14Const*
dtype0*
_output_shapes
: *
valueB
 *  �
�
%training/Adam/clip_by_value_6/MinimumMinimumtraining/Adam/add_17training/Adam/Const_14*
T0*
_output_shapes	
:�
�
training/Adam/clip_by_value_6Maximum%training/Adam/clip_by_value_6/Minimumtraining/Adam/Const_13*
T0*
_output_shapes	
:�
a
training/Adam/Sqrt_6Sqrttraining/Adam/clip_by_value_6*
T0*
_output_shapes	
:�
[
training/Adam/add_18/yConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
q
training/Adam/add_18AddV2training/Adam/Sqrt_6training/Adam/add_18/y*
T0*
_output_shapes	
:�
t
training/Adam/truediv_6RealDivtraining/Adam/mul_30training/Adam/add_18*
T0*
_output_shapes	
:�
i
training/Adam/ReadVariableOp_46ReadVariableOpdense_2/bias*
dtype0*
_output_shapes	
:�
{
training/Adam/sub_19Subtraining/Adam/ReadVariableOp_46training/Adam/truediv_6*
T0*
_output_shapes	
:�
m
!training/Adam/AssignVariableOp_15AssignVariableOptraining/Adam/m_5_1training/Adam/add_16*
dtype0
�
training/Adam/ReadVariableOp_47ReadVariableOptraining/Adam/m_5_1"^training/Adam/AssignVariableOp_15*
dtype0*
_output_shapes	
:�
m
!training/Adam/AssignVariableOp_16AssignVariableOptraining/Adam/v_5_1training/Adam/add_17*
dtype0
�
training/Adam/ReadVariableOp_48ReadVariableOptraining/Adam/v_5_1"^training/Adam/AssignVariableOp_16*
dtype0*
_output_shapes	
:�
f
!training/Adam/AssignVariableOp_17AssignVariableOpdense_2/biastraining/Adam/sub_19*
dtype0
�
training/Adam/ReadVariableOp_49ReadVariableOpdense_2/bias"^training/Adam/AssignVariableOp_17*
dtype0*
_output_shapes	
:�
c
training/Adam/ReadVariableOp_50ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
t
#training/Adam/mul_31/ReadVariableOpReadVariableOptraining/Adam/m_6_1*
dtype0*
_output_shapes	
:�
�
training/Adam/mul_31Multraining/Adam/ReadVariableOp_50#training/Adam/mul_31/ReadVariableOp*
T0*
_output_shapes	
:�
c
training/Adam/ReadVariableOp_51ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
[
training/Adam/sub_20/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
u
training/Adam/sub_20Subtraining/Adam/sub_20/xtraining/Adam/ReadVariableOp_51*
T0*
_output_shapes
: 
x
training/Adam/mul_32Multraining/Adam/sub_20training/Adam/gradients/AddN_15*
T0*
_output_shapes	
:�
o
training/Adam/add_19AddV2training/Adam/mul_31training/Adam/mul_32*
_output_shapes	
:�*
T0
c
training/Adam/ReadVariableOp_52ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
t
#training/Adam/mul_33/ReadVariableOpReadVariableOptraining/Adam/v_6_1*
dtype0*
_output_shapes	
:�
�
training/Adam/mul_33Multraining/Adam/ReadVariableOp_52#training/Adam/mul_33/ReadVariableOp*
T0*
_output_shapes	
:�
c
training/Adam/ReadVariableOp_53ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
[
training/Adam/sub_21/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
u
training/Adam/sub_21Subtraining/Adam/sub_21/xtraining/Adam/ReadVariableOp_53*
T0*
_output_shapes
: 
g
training/Adam/Square_6Squaretraining/Adam/gradients/AddN_15*
_output_shapes	
:�*
T0
o
training/Adam/mul_34Multraining/Adam/sub_21training/Adam/Square_6*
T0*
_output_shapes	
:�
o
training/Adam/add_20AddV2training/Adam/mul_33training/Adam/mul_34*
T0*
_output_shapes	
:�
j
training/Adam/mul_35Multraining/Adam/multraining/Adam/add_19*
T0*
_output_shapes	
:�
[
training/Adam/Const_15Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_16Const*
valueB
 *  �*
dtype0*
_output_shapes
: 
�
%training/Adam/clip_by_value_7/MinimumMinimumtraining/Adam/add_20training/Adam/Const_16*
T0*
_output_shapes	
:�
�
training/Adam/clip_by_value_7Maximum%training/Adam/clip_by_value_7/Minimumtraining/Adam/Const_15*
T0*
_output_shapes	
:�
a
training/Adam/Sqrt_7Sqrttraining/Adam/clip_by_value_7*
T0*
_output_shapes	
:�
[
training/Adam/add_21/yConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
q
training/Adam/add_21AddV2training/Adam/Sqrt_7training/Adam/add_21/y*
T0*
_output_shapes	
:�
t
training/Adam/truediv_7RealDivtraining/Adam/mul_35training/Adam/add_21*
T0*
_output_shapes	
:�
x
training/Adam/ReadVariableOp_54ReadVariableOpbatch_normalization_2/gamma*
dtype0*
_output_shapes	
:�
{
training/Adam/sub_22Subtraining/Adam/ReadVariableOp_54training/Adam/truediv_7*
T0*
_output_shapes	
:�
m
!training/Adam/AssignVariableOp_18AssignVariableOptraining/Adam/m_6_1training/Adam/add_19*
dtype0
�
training/Adam/ReadVariableOp_55ReadVariableOptraining/Adam/m_6_1"^training/Adam/AssignVariableOp_18*
dtype0*
_output_shapes	
:�
m
!training/Adam/AssignVariableOp_19AssignVariableOptraining/Adam/v_6_1training/Adam/add_20*
dtype0
�
training/Adam/ReadVariableOp_56ReadVariableOptraining/Adam/v_6_1"^training/Adam/AssignVariableOp_19*
dtype0*
_output_shapes	
:�
u
!training/Adam/AssignVariableOp_20AssignVariableOpbatch_normalization_2/gammatraining/Adam/sub_22*
dtype0
�
training/Adam/ReadVariableOp_57ReadVariableOpbatch_normalization_2/gamma"^training/Adam/AssignVariableOp_20*
dtype0*
_output_shapes	
:�
c
training/Adam/ReadVariableOp_58ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
t
#training/Adam/mul_36/ReadVariableOpReadVariableOptraining/Adam/m_7_1*
dtype0*
_output_shapes	
:�
�
training/Adam/mul_36Multraining/Adam/ReadVariableOp_58#training/Adam/mul_36/ReadVariableOp*
T0*
_output_shapes	
:�
c
training/Adam/ReadVariableOp_59ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
[
training/Adam/sub_23/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
u
training/Adam/sub_23Subtraining/Adam/sub_23/xtraining/Adam/ReadVariableOp_59*
T0*
_output_shapes
: 
x
training/Adam/mul_37Multraining/Adam/sub_23training/Adam/gradients/AddN_13*
T0*
_output_shapes	
:�
o
training/Adam/add_22AddV2training/Adam/mul_36training/Adam/mul_37*
T0*
_output_shapes	
:�
c
training/Adam/ReadVariableOp_60ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
t
#training/Adam/mul_38/ReadVariableOpReadVariableOptraining/Adam/v_7_1*
dtype0*
_output_shapes	
:�
�
training/Adam/mul_38Multraining/Adam/ReadVariableOp_60#training/Adam/mul_38/ReadVariableOp*
T0*
_output_shapes	
:�
c
training/Adam/ReadVariableOp_61ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
[
training/Adam/sub_24/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
u
training/Adam/sub_24Subtraining/Adam/sub_24/xtraining/Adam/ReadVariableOp_61*
T0*
_output_shapes
: 
g
training/Adam/Square_7Squaretraining/Adam/gradients/AddN_13*
T0*
_output_shapes	
:�
o
training/Adam/mul_39Multraining/Adam/sub_24training/Adam/Square_7*
T0*
_output_shapes	
:�
o
training/Adam/add_23AddV2training/Adam/mul_38training/Adam/mul_39*
_output_shapes	
:�*
T0
j
training/Adam/mul_40Multraining/Adam/multraining/Adam/add_22*
T0*
_output_shapes	
:�
[
training/Adam/Const_17Const*
dtype0*
_output_shapes
: *
valueB
 *    
[
training/Adam/Const_18Const*
valueB
 *  �*
dtype0*
_output_shapes
: 
�
%training/Adam/clip_by_value_8/MinimumMinimumtraining/Adam/add_23training/Adam/Const_18*
T0*
_output_shapes	
:�
�
training/Adam/clip_by_value_8Maximum%training/Adam/clip_by_value_8/Minimumtraining/Adam/Const_17*
T0*
_output_shapes	
:�
a
training/Adam/Sqrt_8Sqrttraining/Adam/clip_by_value_8*
_output_shapes	
:�*
T0
[
training/Adam/add_24/yConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
q
training/Adam/add_24AddV2training/Adam/Sqrt_8training/Adam/add_24/y*
T0*
_output_shapes	
:�
t
training/Adam/truediv_8RealDivtraining/Adam/mul_40training/Adam/add_24*
T0*
_output_shapes	
:�
w
training/Adam/ReadVariableOp_62ReadVariableOpbatch_normalization_2/beta*
dtype0*
_output_shapes	
:�
{
training/Adam/sub_25Subtraining/Adam/ReadVariableOp_62training/Adam/truediv_8*
_output_shapes	
:�*
T0
m
!training/Adam/AssignVariableOp_21AssignVariableOptraining/Adam/m_7_1training/Adam/add_22*
dtype0
�
training/Adam/ReadVariableOp_63ReadVariableOptraining/Adam/m_7_1"^training/Adam/AssignVariableOp_21*
dtype0*
_output_shapes	
:�
m
!training/Adam/AssignVariableOp_22AssignVariableOptraining/Adam/v_7_1training/Adam/add_23*
dtype0
�
training/Adam/ReadVariableOp_64ReadVariableOptraining/Adam/v_7_1"^training/Adam/AssignVariableOp_22*
dtype0*
_output_shapes	
:�
t
!training/Adam/AssignVariableOp_23AssignVariableOpbatch_normalization_2/betatraining/Adam/sub_25*
dtype0
�
training/Adam/ReadVariableOp_65ReadVariableOpbatch_normalization_2/beta"^training/Adam/AssignVariableOp_23*
dtype0*
_output_shapes	
:�
c
training/Adam/ReadVariableOp_66ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
y
#training/Adam/mul_41/ReadVariableOpReadVariableOptraining/Adam/m_8_1*
dtype0* 
_output_shapes
:
��
�
training/Adam/mul_41Multraining/Adam/ReadVariableOp_66#training/Adam/mul_41/ReadVariableOp* 
_output_shapes
:
��*
T0
c
training/Adam/ReadVariableOp_67ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
[
training/Adam/sub_26/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
u
training/Adam/sub_26Subtraining/Adam/sub_26/xtraining/Adam/ReadVariableOp_67*
T0*
_output_shapes
: 
�
training/Adam/mul_42Multraining/Adam/sub_264training/Adam/gradients/dense_3/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:
��
t
training/Adam/add_25AddV2training/Adam/mul_41training/Adam/mul_42* 
_output_shapes
:
��*
T0
c
training/Adam/ReadVariableOp_68ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
y
#training/Adam/mul_43/ReadVariableOpReadVariableOptraining/Adam/v_8_1*
dtype0* 
_output_shapes
:
��
�
training/Adam/mul_43Multraining/Adam/ReadVariableOp_68#training/Adam/mul_43/ReadVariableOp*
T0* 
_output_shapes
:
��
c
training/Adam/ReadVariableOp_69ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
[
training/Adam/sub_27/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
u
training/Adam/sub_27Subtraining/Adam/sub_27/xtraining/Adam/ReadVariableOp_69*
T0*
_output_shapes
: 
�
training/Adam/Square_8Square4training/Adam/gradients/dense_3/MatMul_grad/MatMul_1* 
_output_shapes
:
��*
T0
t
training/Adam/mul_44Multraining/Adam/sub_27training/Adam/Square_8*
T0* 
_output_shapes
:
��
t
training/Adam/add_26AddV2training/Adam/mul_43training/Adam/mul_44*
T0* 
_output_shapes
:
��
o
training/Adam/mul_45Multraining/Adam/multraining/Adam/add_25* 
_output_shapes
:
��*
T0
[
training/Adam/Const_19Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_20Const*
dtype0*
_output_shapes
: *
valueB
 *  �
�
%training/Adam/clip_by_value_9/MinimumMinimumtraining/Adam/add_26training/Adam/Const_20*
T0* 
_output_shapes
:
��
�
training/Adam/clip_by_value_9Maximum%training/Adam/clip_by_value_9/Minimumtraining/Adam/Const_19*
T0* 
_output_shapes
:
��
f
training/Adam/Sqrt_9Sqrttraining/Adam/clip_by_value_9*
T0* 
_output_shapes
:
��
[
training/Adam/add_27/yConst*
dtype0*
_output_shapes
: *
valueB
 *���3
v
training/Adam/add_27AddV2training/Adam/Sqrt_9training/Adam/add_27/y* 
_output_shapes
:
��*
T0
y
training/Adam/truediv_9RealDivtraining/Adam/mul_45training/Adam/add_27* 
_output_shapes
:
��*
T0
p
training/Adam/ReadVariableOp_70ReadVariableOpdense_3/kernel*
dtype0* 
_output_shapes
:
��
�
training/Adam/sub_28Subtraining/Adam/ReadVariableOp_70training/Adam/truediv_9*
T0* 
_output_shapes
:
��
m
!training/Adam/AssignVariableOp_24AssignVariableOptraining/Adam/m_8_1training/Adam/add_25*
dtype0
�
training/Adam/ReadVariableOp_71ReadVariableOptraining/Adam/m_8_1"^training/Adam/AssignVariableOp_24*
dtype0* 
_output_shapes
:
��
m
!training/Adam/AssignVariableOp_25AssignVariableOptraining/Adam/v_8_1training/Adam/add_26*
dtype0
�
training/Adam/ReadVariableOp_72ReadVariableOptraining/Adam/v_8_1"^training/Adam/AssignVariableOp_25*
dtype0* 
_output_shapes
:
��
h
!training/Adam/AssignVariableOp_26AssignVariableOpdense_3/kerneltraining/Adam/sub_28*
dtype0
�
training/Adam/ReadVariableOp_73ReadVariableOpdense_3/kernel"^training/Adam/AssignVariableOp_26*
dtype0* 
_output_shapes
:
��
c
training/Adam/ReadVariableOp_74ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
t
#training/Adam/mul_46/ReadVariableOpReadVariableOptraining/Adam/m_9_1*
dtype0*
_output_shapes	
:�
�
training/Adam/mul_46Multraining/Adam/ReadVariableOp_74#training/Adam/mul_46/ReadVariableOp*
_output_shapes	
:�*
T0
c
training/Adam/ReadVariableOp_75ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
[
training/Adam/sub_29/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
u
training/Adam/sub_29Subtraining/Adam/sub_29/xtraining/Adam/ReadVariableOp_75*
_output_shapes
: *
T0
�
training/Adam/mul_47Multraining/Adam/sub_298training/Adam/gradients/dense_3/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:�
o
training/Adam/add_28AddV2training/Adam/mul_46training/Adam/mul_47*
T0*
_output_shapes	
:�
c
training/Adam/ReadVariableOp_76ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
t
#training/Adam/mul_48/ReadVariableOpReadVariableOptraining/Adam/v_9_1*
dtype0*
_output_shapes	
:�
�
training/Adam/mul_48Multraining/Adam/ReadVariableOp_76#training/Adam/mul_48/ReadVariableOp*
T0*
_output_shapes	
:�
c
training/Adam/ReadVariableOp_77ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
[
training/Adam/sub_30/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
u
training/Adam/sub_30Subtraining/Adam/sub_30/xtraining/Adam/ReadVariableOp_77*
T0*
_output_shapes
: 
�
training/Adam/Square_9Square8training/Adam/gradients/dense_3/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:�
o
training/Adam/mul_49Multraining/Adam/sub_30training/Adam/Square_9*
T0*
_output_shapes	
:�
o
training/Adam/add_29AddV2training/Adam/mul_48training/Adam/mul_49*
T0*
_output_shapes	
:�
j
training/Adam/mul_50Multraining/Adam/multraining/Adam/add_28*
T0*
_output_shapes	
:�
[
training/Adam/Const_21Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_22Const*
valueB
 *  �*
dtype0*
_output_shapes
: 
�
&training/Adam/clip_by_value_10/MinimumMinimumtraining/Adam/add_29training/Adam/Const_22*
T0*
_output_shapes	
:�
�
training/Adam/clip_by_value_10Maximum&training/Adam/clip_by_value_10/Minimumtraining/Adam/Const_21*
T0*
_output_shapes	
:�
c
training/Adam/Sqrt_10Sqrttraining/Adam/clip_by_value_10*
T0*
_output_shapes	
:�
[
training/Adam/add_30/yConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
r
training/Adam/add_30AddV2training/Adam/Sqrt_10training/Adam/add_30/y*
T0*
_output_shapes	
:�
u
training/Adam/truediv_10RealDivtraining/Adam/mul_50training/Adam/add_30*
T0*
_output_shapes	
:�
i
training/Adam/ReadVariableOp_78ReadVariableOpdense_3/bias*
dtype0*
_output_shapes	
:�
|
training/Adam/sub_31Subtraining/Adam/ReadVariableOp_78training/Adam/truediv_10*
T0*
_output_shapes	
:�
m
!training/Adam/AssignVariableOp_27AssignVariableOptraining/Adam/m_9_1training/Adam/add_28*
dtype0
�
training/Adam/ReadVariableOp_79ReadVariableOptraining/Adam/m_9_1"^training/Adam/AssignVariableOp_27*
dtype0*
_output_shapes	
:�
m
!training/Adam/AssignVariableOp_28AssignVariableOptraining/Adam/v_9_1training/Adam/add_29*
dtype0
�
training/Adam/ReadVariableOp_80ReadVariableOptraining/Adam/v_9_1"^training/Adam/AssignVariableOp_28*
dtype0*
_output_shapes	
:�
f
!training/Adam/AssignVariableOp_29AssignVariableOpdense_3/biastraining/Adam/sub_31*
dtype0
�
training/Adam/ReadVariableOp_81ReadVariableOpdense_3/bias"^training/Adam/AssignVariableOp_29*
dtype0*
_output_shapes	
:�
c
training/Adam/ReadVariableOp_82ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
u
#training/Adam/mul_51/ReadVariableOpReadVariableOptraining/Adam/m_10_1*
dtype0*
_output_shapes	
:�
�
training/Adam/mul_51Multraining/Adam/ReadVariableOp_82#training/Adam/mul_51/ReadVariableOp*
T0*
_output_shapes	
:�
c
training/Adam/ReadVariableOp_83ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
[
training/Adam/sub_32/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
u
training/Adam/sub_32Subtraining/Adam/sub_32/xtraining/Adam/ReadVariableOp_83*
T0*
_output_shapes
: 
w
training/Adam/mul_52Multraining/Adam/sub_32training/Adam/gradients/AddN_9*
T0*
_output_shapes	
:�
o
training/Adam/add_31AddV2training/Adam/mul_51training/Adam/mul_52*
_output_shapes	
:�*
T0
c
training/Adam/ReadVariableOp_84ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
u
#training/Adam/mul_53/ReadVariableOpReadVariableOptraining/Adam/v_10_1*
dtype0*
_output_shapes	
:�
�
training/Adam/mul_53Multraining/Adam/ReadVariableOp_84#training/Adam/mul_53/ReadVariableOp*
T0*
_output_shapes	
:�
c
training/Adam/ReadVariableOp_85ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
[
training/Adam/sub_33/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
u
training/Adam/sub_33Subtraining/Adam/sub_33/xtraining/Adam/ReadVariableOp_85*
T0*
_output_shapes
: 
g
training/Adam/Square_10Squaretraining/Adam/gradients/AddN_9*
T0*
_output_shapes	
:�
p
training/Adam/mul_54Multraining/Adam/sub_33training/Adam/Square_10*
T0*
_output_shapes	
:�
o
training/Adam/add_32AddV2training/Adam/mul_53training/Adam/mul_54*
T0*
_output_shapes	
:�
j
training/Adam/mul_55Multraining/Adam/multraining/Adam/add_31*
T0*
_output_shapes	
:�
[
training/Adam/Const_23Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_24Const*
dtype0*
_output_shapes
: *
valueB
 *  �
�
&training/Adam/clip_by_value_11/MinimumMinimumtraining/Adam/add_32training/Adam/Const_24*
_output_shapes	
:�*
T0
�
training/Adam/clip_by_value_11Maximum&training/Adam/clip_by_value_11/Minimumtraining/Adam/Const_23*
T0*
_output_shapes	
:�
c
training/Adam/Sqrt_11Sqrttraining/Adam/clip_by_value_11*
T0*
_output_shapes	
:�
[
training/Adam/add_33/yConst*
dtype0*
_output_shapes
: *
valueB
 *���3
r
training/Adam/add_33AddV2training/Adam/Sqrt_11training/Adam/add_33/y*
_output_shapes	
:�*
T0
u
training/Adam/truediv_11RealDivtraining/Adam/mul_55training/Adam/add_33*
T0*
_output_shapes	
:�
x
training/Adam/ReadVariableOp_86ReadVariableOpbatch_normalization_3/gamma*
dtype0*
_output_shapes	
:�
|
training/Adam/sub_34Subtraining/Adam/ReadVariableOp_86training/Adam/truediv_11*
_output_shapes	
:�*
T0
n
!training/Adam/AssignVariableOp_30AssignVariableOptraining/Adam/m_10_1training/Adam/add_31*
dtype0
�
training/Adam/ReadVariableOp_87ReadVariableOptraining/Adam/m_10_1"^training/Adam/AssignVariableOp_30*
dtype0*
_output_shapes	
:�
n
!training/Adam/AssignVariableOp_31AssignVariableOptraining/Adam/v_10_1training/Adam/add_32*
dtype0
�
training/Adam/ReadVariableOp_88ReadVariableOptraining/Adam/v_10_1"^training/Adam/AssignVariableOp_31*
dtype0*
_output_shapes	
:�
u
!training/Adam/AssignVariableOp_32AssignVariableOpbatch_normalization_3/gammatraining/Adam/sub_34*
dtype0
�
training/Adam/ReadVariableOp_89ReadVariableOpbatch_normalization_3/gamma"^training/Adam/AssignVariableOp_32*
dtype0*
_output_shapes	
:�
c
training/Adam/ReadVariableOp_90ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
u
#training/Adam/mul_56/ReadVariableOpReadVariableOptraining/Adam/m_11_1*
dtype0*
_output_shapes	
:�
�
training/Adam/mul_56Multraining/Adam/ReadVariableOp_90#training/Adam/mul_56/ReadVariableOp*
T0*
_output_shapes	
:�
c
training/Adam/ReadVariableOp_91ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
[
training/Adam/sub_35/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
u
training/Adam/sub_35Subtraining/Adam/sub_35/xtraining/Adam/ReadVariableOp_91*
T0*
_output_shapes
: 
w
training/Adam/mul_57Multraining/Adam/sub_35training/Adam/gradients/AddN_7*
T0*
_output_shapes	
:�
o
training/Adam/add_34AddV2training/Adam/mul_56training/Adam/mul_57*
T0*
_output_shapes	
:�
c
training/Adam/ReadVariableOp_92ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
u
#training/Adam/mul_58/ReadVariableOpReadVariableOptraining/Adam/v_11_1*
dtype0*
_output_shapes	
:�
�
training/Adam/mul_58Multraining/Adam/ReadVariableOp_92#training/Adam/mul_58/ReadVariableOp*
T0*
_output_shapes	
:�
c
training/Adam/ReadVariableOp_93ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
[
training/Adam/sub_36/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
u
training/Adam/sub_36Subtraining/Adam/sub_36/xtraining/Adam/ReadVariableOp_93*
T0*
_output_shapes
: 
g
training/Adam/Square_11Squaretraining/Adam/gradients/AddN_7*
T0*
_output_shapes	
:�
p
training/Adam/mul_59Multraining/Adam/sub_36training/Adam/Square_11*
T0*
_output_shapes	
:�
o
training/Adam/add_35AddV2training/Adam/mul_58training/Adam/mul_59*
T0*
_output_shapes	
:�
j
training/Adam/mul_60Multraining/Adam/multraining/Adam/add_34*
T0*
_output_shapes	
:�
[
training/Adam/Const_25Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_26Const*
valueB
 *  �*
dtype0*
_output_shapes
: 
�
&training/Adam/clip_by_value_12/MinimumMinimumtraining/Adam/add_35training/Adam/Const_26*
T0*
_output_shapes	
:�
�
training/Adam/clip_by_value_12Maximum&training/Adam/clip_by_value_12/Minimumtraining/Adam/Const_25*
_output_shapes	
:�*
T0
c
training/Adam/Sqrt_12Sqrttraining/Adam/clip_by_value_12*
T0*
_output_shapes	
:�
[
training/Adam/add_36/yConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
r
training/Adam/add_36AddV2training/Adam/Sqrt_12training/Adam/add_36/y*
T0*
_output_shapes	
:�
u
training/Adam/truediv_12RealDivtraining/Adam/mul_60training/Adam/add_36*
T0*
_output_shapes	
:�
w
training/Adam/ReadVariableOp_94ReadVariableOpbatch_normalization_3/beta*
dtype0*
_output_shapes	
:�
|
training/Adam/sub_37Subtraining/Adam/ReadVariableOp_94training/Adam/truediv_12*
_output_shapes	
:�*
T0
n
!training/Adam/AssignVariableOp_33AssignVariableOptraining/Adam/m_11_1training/Adam/add_34*
dtype0
�
training/Adam/ReadVariableOp_95ReadVariableOptraining/Adam/m_11_1"^training/Adam/AssignVariableOp_33*
dtype0*
_output_shapes	
:�
n
!training/Adam/AssignVariableOp_34AssignVariableOptraining/Adam/v_11_1training/Adam/add_35*
dtype0
�
training/Adam/ReadVariableOp_96ReadVariableOptraining/Adam/v_11_1"^training/Adam/AssignVariableOp_34*
dtype0*
_output_shapes	
:�
t
!training/Adam/AssignVariableOp_35AssignVariableOpbatch_normalization_3/betatraining/Adam/sub_37*
dtype0
�
training/Adam/ReadVariableOp_97ReadVariableOpbatch_normalization_3/beta"^training/Adam/AssignVariableOp_35*
dtype0*
_output_shapes	
:�
c
training/Adam/ReadVariableOp_98ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
z
#training/Adam/mul_61/ReadVariableOpReadVariableOptraining/Adam/m_12_1*
dtype0* 
_output_shapes
:
��
�
training/Adam/mul_61Multraining/Adam/ReadVariableOp_98#training/Adam/mul_61/ReadVariableOp* 
_output_shapes
:
��*
T0
c
training/Adam/ReadVariableOp_99ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
[
training/Adam/sub_38/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
u
training/Adam/sub_38Subtraining/Adam/sub_38/xtraining/Adam/ReadVariableOp_99*
T0*
_output_shapes
: 
�
training/Adam/mul_62Multraining/Adam/sub_384training/Adam/gradients/dense_4/MatMul_grad/MatMul_1* 
_output_shapes
:
��*
T0
t
training/Adam/add_37AddV2training/Adam/mul_61training/Adam/mul_62*
T0* 
_output_shapes
:
��
d
 training/Adam/ReadVariableOp_100ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
z
#training/Adam/mul_63/ReadVariableOpReadVariableOptraining/Adam/v_12_1*
dtype0* 
_output_shapes
:
��
�
training/Adam/mul_63Mul training/Adam/ReadVariableOp_100#training/Adam/mul_63/ReadVariableOp*
T0* 
_output_shapes
:
��
d
 training/Adam/ReadVariableOp_101ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
[
training/Adam/sub_39/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
v
training/Adam/sub_39Subtraining/Adam/sub_39/x training/Adam/ReadVariableOp_101*
_output_shapes
: *
T0
�
training/Adam/Square_12Square4training/Adam/gradients/dense_4/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:
��
u
training/Adam/mul_64Multraining/Adam/sub_39training/Adam/Square_12*
T0* 
_output_shapes
:
��
t
training/Adam/add_38AddV2training/Adam/mul_63training/Adam/mul_64*
T0* 
_output_shapes
:
��
o
training/Adam/mul_65Multraining/Adam/multraining/Adam/add_37* 
_output_shapes
:
��*
T0
[
training/Adam/Const_27Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_28Const*
dtype0*
_output_shapes
: *
valueB
 *  �
�
&training/Adam/clip_by_value_13/MinimumMinimumtraining/Adam/add_38training/Adam/Const_28*
T0* 
_output_shapes
:
��
�
training/Adam/clip_by_value_13Maximum&training/Adam/clip_by_value_13/Minimumtraining/Adam/Const_27* 
_output_shapes
:
��*
T0
h
training/Adam/Sqrt_13Sqrttraining/Adam/clip_by_value_13*
T0* 
_output_shapes
:
��
[
training/Adam/add_39/yConst*
dtype0*
_output_shapes
: *
valueB
 *���3
w
training/Adam/add_39AddV2training/Adam/Sqrt_13training/Adam/add_39/y*
T0* 
_output_shapes
:
��
z
training/Adam/truediv_13RealDivtraining/Adam/mul_65training/Adam/add_39*
T0* 
_output_shapes
:
��
q
 training/Adam/ReadVariableOp_102ReadVariableOpdense_4/kernel*
dtype0* 
_output_shapes
:
��
�
training/Adam/sub_40Sub training/Adam/ReadVariableOp_102training/Adam/truediv_13*
T0* 
_output_shapes
:
��
n
!training/Adam/AssignVariableOp_36AssignVariableOptraining/Adam/m_12_1training/Adam/add_37*
dtype0
�
 training/Adam/ReadVariableOp_103ReadVariableOptraining/Adam/m_12_1"^training/Adam/AssignVariableOp_36*
dtype0* 
_output_shapes
:
��
n
!training/Adam/AssignVariableOp_37AssignVariableOptraining/Adam/v_12_1training/Adam/add_38*
dtype0
�
 training/Adam/ReadVariableOp_104ReadVariableOptraining/Adam/v_12_1"^training/Adam/AssignVariableOp_37*
dtype0* 
_output_shapes
:
��
h
!training/Adam/AssignVariableOp_38AssignVariableOpdense_4/kerneltraining/Adam/sub_40*
dtype0
�
 training/Adam/ReadVariableOp_105ReadVariableOpdense_4/kernel"^training/Adam/AssignVariableOp_38*
dtype0* 
_output_shapes
:
��
d
 training/Adam/ReadVariableOp_106ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
u
#training/Adam/mul_66/ReadVariableOpReadVariableOptraining/Adam/m_13_1*
dtype0*
_output_shapes	
:�
�
training/Adam/mul_66Mul training/Adam/ReadVariableOp_106#training/Adam/mul_66/ReadVariableOp*
_output_shapes	
:�*
T0
d
 training/Adam/ReadVariableOp_107ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
[
training/Adam/sub_41/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
v
training/Adam/sub_41Subtraining/Adam/sub_41/x training/Adam/ReadVariableOp_107*
T0*
_output_shapes
: 
�
training/Adam/mul_67Multraining/Adam/sub_418training/Adam/gradients/dense_4/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:�
o
training/Adam/add_40AddV2training/Adam/mul_66training/Adam/mul_67*
T0*
_output_shapes	
:�
d
 training/Adam/ReadVariableOp_108ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
u
#training/Adam/mul_68/ReadVariableOpReadVariableOptraining/Adam/v_13_1*
dtype0*
_output_shapes	
:�
�
training/Adam/mul_68Mul training/Adam/ReadVariableOp_108#training/Adam/mul_68/ReadVariableOp*
T0*
_output_shapes	
:�
d
 training/Adam/ReadVariableOp_109ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
[
training/Adam/sub_42/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
v
training/Adam/sub_42Subtraining/Adam/sub_42/x training/Adam/ReadVariableOp_109*
_output_shapes
: *
T0
�
training/Adam/Square_13Square8training/Adam/gradients/dense_4/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:�
p
training/Adam/mul_69Multraining/Adam/sub_42training/Adam/Square_13*
T0*
_output_shapes	
:�
o
training/Adam/add_41AddV2training/Adam/mul_68training/Adam/mul_69*
_output_shapes	
:�*
T0
j
training/Adam/mul_70Multraining/Adam/multraining/Adam/add_40*
T0*
_output_shapes	
:�
[
training/Adam/Const_29Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_30Const*
valueB
 *  �*
dtype0*
_output_shapes
: 
�
&training/Adam/clip_by_value_14/MinimumMinimumtraining/Adam/add_41training/Adam/Const_30*
_output_shapes	
:�*
T0
�
training/Adam/clip_by_value_14Maximum&training/Adam/clip_by_value_14/Minimumtraining/Adam/Const_29*
_output_shapes	
:�*
T0
c
training/Adam/Sqrt_14Sqrttraining/Adam/clip_by_value_14*
_output_shapes	
:�*
T0
[
training/Adam/add_42/yConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
r
training/Adam/add_42AddV2training/Adam/Sqrt_14training/Adam/add_42/y*
T0*
_output_shapes	
:�
u
training/Adam/truediv_14RealDivtraining/Adam/mul_70training/Adam/add_42*
_output_shapes	
:�*
T0
j
 training/Adam/ReadVariableOp_110ReadVariableOpdense_4/bias*
dtype0*
_output_shapes	
:�
}
training/Adam/sub_43Sub training/Adam/ReadVariableOp_110training/Adam/truediv_14*
T0*
_output_shapes	
:�
n
!training/Adam/AssignVariableOp_39AssignVariableOptraining/Adam/m_13_1training/Adam/add_40*
dtype0
�
 training/Adam/ReadVariableOp_111ReadVariableOptraining/Adam/m_13_1"^training/Adam/AssignVariableOp_39*
dtype0*
_output_shapes	
:�
n
!training/Adam/AssignVariableOp_40AssignVariableOptraining/Adam/v_13_1training/Adam/add_41*
dtype0
�
 training/Adam/ReadVariableOp_112ReadVariableOptraining/Adam/v_13_1"^training/Adam/AssignVariableOp_40*
dtype0*
_output_shapes	
:�
f
!training/Adam/AssignVariableOp_41AssignVariableOpdense_4/biastraining/Adam/sub_43*
dtype0
�
 training/Adam/ReadVariableOp_113ReadVariableOpdense_4/bias"^training/Adam/AssignVariableOp_41*
dtype0*
_output_shapes	
:�
d
 training/Adam/ReadVariableOp_114ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
u
#training/Adam/mul_71/ReadVariableOpReadVariableOptraining/Adam/m_14_1*
dtype0*
_output_shapes	
:�
�
training/Adam/mul_71Mul training/Adam/ReadVariableOp_114#training/Adam/mul_71/ReadVariableOp*
T0*
_output_shapes	
:�
d
 training/Adam/ReadVariableOp_115ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
[
training/Adam/sub_44/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
v
training/Adam/sub_44Subtraining/Adam/sub_44/x training/Adam/ReadVariableOp_115*
T0*
_output_shapes
: 
w
training/Adam/mul_72Multraining/Adam/sub_44training/Adam/gradients/AddN_3*
T0*
_output_shapes	
:�
o
training/Adam/add_43AddV2training/Adam/mul_71training/Adam/mul_72*
T0*
_output_shapes	
:�
d
 training/Adam/ReadVariableOp_116ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
u
#training/Adam/mul_73/ReadVariableOpReadVariableOptraining/Adam/v_14_1*
dtype0*
_output_shapes	
:�
�
training/Adam/mul_73Mul training/Adam/ReadVariableOp_116#training/Adam/mul_73/ReadVariableOp*
_output_shapes	
:�*
T0
d
 training/Adam/ReadVariableOp_117ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
[
training/Adam/sub_45/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
v
training/Adam/sub_45Subtraining/Adam/sub_45/x training/Adam/ReadVariableOp_117*
T0*
_output_shapes
: 
g
training/Adam/Square_14Squaretraining/Adam/gradients/AddN_3*
T0*
_output_shapes	
:�
p
training/Adam/mul_74Multraining/Adam/sub_45training/Adam/Square_14*
T0*
_output_shapes	
:�
o
training/Adam/add_44AddV2training/Adam/mul_73training/Adam/mul_74*
T0*
_output_shapes	
:�
j
training/Adam/mul_75Multraining/Adam/multraining/Adam/add_43*
T0*
_output_shapes	
:�
[
training/Adam/Const_31Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_32Const*
valueB
 *  �*
dtype0*
_output_shapes
: 
�
&training/Adam/clip_by_value_15/MinimumMinimumtraining/Adam/add_44training/Adam/Const_32*
T0*
_output_shapes	
:�
�
training/Adam/clip_by_value_15Maximum&training/Adam/clip_by_value_15/Minimumtraining/Adam/Const_31*
_output_shapes	
:�*
T0
c
training/Adam/Sqrt_15Sqrttraining/Adam/clip_by_value_15*
T0*
_output_shapes	
:�
[
training/Adam/add_45/yConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
r
training/Adam/add_45AddV2training/Adam/Sqrt_15training/Adam/add_45/y*
T0*
_output_shapes	
:�
u
training/Adam/truediv_15RealDivtraining/Adam/mul_75training/Adam/add_45*
_output_shapes	
:�*
T0
y
 training/Adam/ReadVariableOp_118ReadVariableOpbatch_normalization_4/gamma*
dtype0*
_output_shapes	
:�
}
training/Adam/sub_46Sub training/Adam/ReadVariableOp_118training/Adam/truediv_15*
T0*
_output_shapes	
:�
n
!training/Adam/AssignVariableOp_42AssignVariableOptraining/Adam/m_14_1training/Adam/add_43*
dtype0
�
 training/Adam/ReadVariableOp_119ReadVariableOptraining/Adam/m_14_1"^training/Adam/AssignVariableOp_42*
dtype0*
_output_shapes	
:�
n
!training/Adam/AssignVariableOp_43AssignVariableOptraining/Adam/v_14_1training/Adam/add_44*
dtype0
�
 training/Adam/ReadVariableOp_120ReadVariableOptraining/Adam/v_14_1"^training/Adam/AssignVariableOp_43*
dtype0*
_output_shapes	
:�
u
!training/Adam/AssignVariableOp_44AssignVariableOpbatch_normalization_4/gammatraining/Adam/sub_46*
dtype0
�
 training/Adam/ReadVariableOp_121ReadVariableOpbatch_normalization_4/gamma"^training/Adam/AssignVariableOp_44*
dtype0*
_output_shapes	
:�
d
 training/Adam/ReadVariableOp_122ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
u
#training/Adam/mul_76/ReadVariableOpReadVariableOptraining/Adam/m_15_1*
dtype0*
_output_shapes	
:�
�
training/Adam/mul_76Mul training/Adam/ReadVariableOp_122#training/Adam/mul_76/ReadVariableOp*
T0*
_output_shapes	
:�
d
 training/Adam/ReadVariableOp_123ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
[
training/Adam/sub_47/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
v
training/Adam/sub_47Subtraining/Adam/sub_47/x training/Adam/ReadVariableOp_123*
_output_shapes
: *
T0
w
training/Adam/mul_77Multraining/Adam/sub_47training/Adam/gradients/AddN_1*
T0*
_output_shapes	
:�
o
training/Adam/add_46AddV2training/Adam/mul_76training/Adam/mul_77*
T0*
_output_shapes	
:�
d
 training/Adam/ReadVariableOp_124ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
u
#training/Adam/mul_78/ReadVariableOpReadVariableOptraining/Adam/v_15_1*
dtype0*
_output_shapes	
:�
�
training/Adam/mul_78Mul training/Adam/ReadVariableOp_124#training/Adam/mul_78/ReadVariableOp*
_output_shapes	
:�*
T0
d
 training/Adam/ReadVariableOp_125ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
[
training/Adam/sub_48/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
v
training/Adam/sub_48Subtraining/Adam/sub_48/x training/Adam/ReadVariableOp_125*
_output_shapes
: *
T0
g
training/Adam/Square_15Squaretraining/Adam/gradients/AddN_1*
T0*
_output_shapes	
:�
p
training/Adam/mul_79Multraining/Adam/sub_48training/Adam/Square_15*
T0*
_output_shapes	
:�
o
training/Adam/add_47AddV2training/Adam/mul_78training/Adam/mul_79*
_output_shapes	
:�*
T0
j
training/Adam/mul_80Multraining/Adam/multraining/Adam/add_46*
T0*
_output_shapes	
:�
[
training/Adam/Const_33Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_34Const*
valueB
 *  �*
dtype0*
_output_shapes
: 
�
&training/Adam/clip_by_value_16/MinimumMinimumtraining/Adam/add_47training/Adam/Const_34*
T0*
_output_shapes	
:�
�
training/Adam/clip_by_value_16Maximum&training/Adam/clip_by_value_16/Minimumtraining/Adam/Const_33*
T0*
_output_shapes	
:�
c
training/Adam/Sqrt_16Sqrttraining/Adam/clip_by_value_16*
_output_shapes	
:�*
T0
[
training/Adam/add_48/yConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
r
training/Adam/add_48AddV2training/Adam/Sqrt_16training/Adam/add_48/y*
T0*
_output_shapes	
:�
u
training/Adam/truediv_16RealDivtraining/Adam/mul_80training/Adam/add_48*
T0*
_output_shapes	
:�
x
 training/Adam/ReadVariableOp_126ReadVariableOpbatch_normalization_4/beta*
dtype0*
_output_shapes	
:�
}
training/Adam/sub_49Sub training/Adam/ReadVariableOp_126training/Adam/truediv_16*
_output_shapes	
:�*
T0
n
!training/Adam/AssignVariableOp_45AssignVariableOptraining/Adam/m_15_1training/Adam/add_46*
dtype0
�
 training/Adam/ReadVariableOp_127ReadVariableOptraining/Adam/m_15_1"^training/Adam/AssignVariableOp_45*
dtype0*
_output_shapes	
:�
n
!training/Adam/AssignVariableOp_46AssignVariableOptraining/Adam/v_15_1training/Adam/add_47*
dtype0
�
 training/Adam/ReadVariableOp_128ReadVariableOptraining/Adam/v_15_1"^training/Adam/AssignVariableOp_46*
dtype0*
_output_shapes	
:�
t
!training/Adam/AssignVariableOp_47AssignVariableOpbatch_normalization_4/betatraining/Adam/sub_49*
dtype0
�
 training/Adam/ReadVariableOp_129ReadVariableOpbatch_normalization_4/beta"^training/Adam/AssignVariableOp_47*
dtype0*
_output_shapes	
:�
d
 training/Adam/ReadVariableOp_130ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
y
#training/Adam/mul_81/ReadVariableOpReadVariableOptraining/Adam/m_16_1*
dtype0*
_output_shapes
:	�+
�
training/Adam/mul_81Mul training/Adam/ReadVariableOp_130#training/Adam/mul_81/ReadVariableOp*
_output_shapes
:	�+*
T0
d
 training/Adam/ReadVariableOp_131ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
[
training/Adam/sub_50/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
v
training/Adam/sub_50Subtraining/Adam/sub_50/x training/Adam/ReadVariableOp_131*
T0*
_output_shapes
: 
�
training/Adam/mul_82Multraining/Adam/sub_504training/Adam/gradients/dense_5/MatMul_grad/MatMul_1*
T0*
_output_shapes
:	�+
s
training/Adam/add_49AddV2training/Adam/mul_81training/Adam/mul_82*
T0*
_output_shapes
:	�+
d
 training/Adam/ReadVariableOp_132ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
y
#training/Adam/mul_83/ReadVariableOpReadVariableOptraining/Adam/v_16_1*
dtype0*
_output_shapes
:	�+
�
training/Adam/mul_83Mul training/Adam/ReadVariableOp_132#training/Adam/mul_83/ReadVariableOp*
T0*
_output_shapes
:	�+
d
 training/Adam/ReadVariableOp_133ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
[
training/Adam/sub_51/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
v
training/Adam/sub_51Subtraining/Adam/sub_51/x training/Adam/ReadVariableOp_133*
_output_shapes
: *
T0
�
training/Adam/Square_16Square4training/Adam/gradients/dense_5/MatMul_grad/MatMul_1*
T0*
_output_shapes
:	�+
t
training/Adam/mul_84Multraining/Adam/sub_51training/Adam/Square_16*
T0*
_output_shapes
:	�+
s
training/Adam/add_50AddV2training/Adam/mul_83training/Adam/mul_84*
T0*
_output_shapes
:	�+
n
training/Adam/mul_85Multraining/Adam/multraining/Adam/add_49*
T0*
_output_shapes
:	�+
[
training/Adam/Const_35Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_36Const*
valueB
 *  �*
dtype0*
_output_shapes
: 
�
&training/Adam/clip_by_value_17/MinimumMinimumtraining/Adam/add_50training/Adam/Const_36*
_output_shapes
:	�+*
T0
�
training/Adam/clip_by_value_17Maximum&training/Adam/clip_by_value_17/Minimumtraining/Adam/Const_35*
T0*
_output_shapes
:	�+
g
training/Adam/Sqrt_17Sqrttraining/Adam/clip_by_value_17*
T0*
_output_shapes
:	�+
[
training/Adam/add_51/yConst*
dtype0*
_output_shapes
: *
valueB
 *���3
v
training/Adam/add_51AddV2training/Adam/Sqrt_17training/Adam/add_51/y*
T0*
_output_shapes
:	�+
y
training/Adam/truediv_17RealDivtraining/Adam/mul_85training/Adam/add_51*
T0*
_output_shapes
:	�+
p
 training/Adam/ReadVariableOp_134ReadVariableOpdense_5/kernel*
dtype0*
_output_shapes
:	�+
�
training/Adam/sub_52Sub training/Adam/ReadVariableOp_134training/Adam/truediv_17*
T0*
_output_shapes
:	�+
n
!training/Adam/AssignVariableOp_48AssignVariableOptraining/Adam/m_16_1training/Adam/add_49*
dtype0
�
 training/Adam/ReadVariableOp_135ReadVariableOptraining/Adam/m_16_1"^training/Adam/AssignVariableOp_48*
dtype0*
_output_shapes
:	�+
n
!training/Adam/AssignVariableOp_49AssignVariableOptraining/Adam/v_16_1training/Adam/add_50*
dtype0
�
 training/Adam/ReadVariableOp_136ReadVariableOptraining/Adam/v_16_1"^training/Adam/AssignVariableOp_49*
dtype0*
_output_shapes
:	�+
h
!training/Adam/AssignVariableOp_50AssignVariableOpdense_5/kerneltraining/Adam/sub_52*
dtype0
�
 training/Adam/ReadVariableOp_137ReadVariableOpdense_5/kernel"^training/Adam/AssignVariableOp_50*
dtype0*
_output_shapes
:	�+
d
 training/Adam/ReadVariableOp_138ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
t
#training/Adam/mul_86/ReadVariableOpReadVariableOptraining/Adam/m_17_1*
dtype0*
_output_shapes
:+
�
training/Adam/mul_86Mul training/Adam/ReadVariableOp_138#training/Adam/mul_86/ReadVariableOp*
T0*
_output_shapes
:+
d
 training/Adam/ReadVariableOp_139ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
[
training/Adam/sub_53/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
v
training/Adam/sub_53Subtraining/Adam/sub_53/x training/Adam/ReadVariableOp_139*
_output_shapes
: *
T0
�
training/Adam/mul_87Multraining/Adam/sub_538training/Adam/gradients/dense_5/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:+
n
training/Adam/add_52AddV2training/Adam/mul_86training/Adam/mul_87*
_output_shapes
:+*
T0
d
 training/Adam/ReadVariableOp_140ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
t
#training/Adam/mul_88/ReadVariableOpReadVariableOptraining/Adam/v_17_1*
dtype0*
_output_shapes
:+
�
training/Adam/mul_88Mul training/Adam/ReadVariableOp_140#training/Adam/mul_88/ReadVariableOp*
T0*
_output_shapes
:+
d
 training/Adam/ReadVariableOp_141ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
[
training/Adam/sub_54/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
v
training/Adam/sub_54Subtraining/Adam/sub_54/x training/Adam/ReadVariableOp_141*
_output_shapes
: *
T0
�
training/Adam/Square_17Square8training/Adam/gradients/dense_5/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:+
o
training/Adam/mul_89Multraining/Adam/sub_54training/Adam/Square_17*
T0*
_output_shapes
:+
n
training/Adam/add_53AddV2training/Adam/mul_88training/Adam/mul_89*
T0*
_output_shapes
:+
i
training/Adam/mul_90Multraining/Adam/multraining/Adam/add_52*
T0*
_output_shapes
:+
[
training/Adam/Const_37Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_38Const*
valueB
 *  �*
dtype0*
_output_shapes
: 
�
&training/Adam/clip_by_value_18/MinimumMinimumtraining/Adam/add_53training/Adam/Const_38*
T0*
_output_shapes
:+
�
training/Adam/clip_by_value_18Maximum&training/Adam/clip_by_value_18/Minimumtraining/Adam/Const_37*
T0*
_output_shapes
:+
b
training/Adam/Sqrt_18Sqrttraining/Adam/clip_by_value_18*
T0*
_output_shapes
:+
[
training/Adam/add_54/yConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
q
training/Adam/add_54AddV2training/Adam/Sqrt_18training/Adam/add_54/y*
T0*
_output_shapes
:+
t
training/Adam/truediv_18RealDivtraining/Adam/mul_90training/Adam/add_54*
T0*
_output_shapes
:+
i
 training/Adam/ReadVariableOp_142ReadVariableOpdense_5/bias*
dtype0*
_output_shapes
:+
|
training/Adam/sub_55Sub training/Adam/ReadVariableOp_142training/Adam/truediv_18*
T0*
_output_shapes
:+
n
!training/Adam/AssignVariableOp_51AssignVariableOptraining/Adam/m_17_1training/Adam/add_52*
dtype0
�
 training/Adam/ReadVariableOp_143ReadVariableOptraining/Adam/m_17_1"^training/Adam/AssignVariableOp_51*
dtype0*
_output_shapes
:+
n
!training/Adam/AssignVariableOp_52AssignVariableOptraining/Adam/v_17_1training/Adam/add_53*
dtype0
�
 training/Adam/ReadVariableOp_144ReadVariableOptraining/Adam/v_17_1"^training/Adam/AssignVariableOp_52*
dtype0*
_output_shapes
:+
f
!training/Adam/AssignVariableOp_53AssignVariableOpdense_5/biastraining/Adam/sub_55*
dtype0
�
 training/Adam/ReadVariableOp_145ReadVariableOpdense_5/bias"^training/Adam/AssignVariableOp_53*
dtype0*
_output_shapes
:+
W
training/VarIsInitializedOpVarIsInitializedOpdense_1/bias*
_output_shapes
: 
R
training/VarIsInitializedOp_1VarIsInitializedOpcount*
_output_shapes
: 
`
training/VarIsInitializedOp_2VarIsInitializedOptraining/Adam/v_5_1*
_output_shapes
: 
a
training/VarIsInitializedOp_3VarIsInitializedOptraining/Adam/m_15_1*
_output_shapes
: 
d
training/VarIsInitializedOp_4VarIsInitializedOptraining/Adam/vhat_12_1*
_output_shapes
: 
a
training/VarIsInitializedOp_5VarIsInitializedOptraining/Adam/m_14_1*
_output_shapes
: 
n
training/VarIsInitializedOp_6VarIsInitializedOp!batch_normalization_3/moving_mean*
_output_shapes
: 
`
training/VarIsInitializedOp_7VarIsInitializedOptraining/Adam/m_3_1*
_output_shapes
: 
`
training/VarIsInitializedOp_8VarIsInitializedOptraining/Adam/m_8_1*
_output_shapes
: 
a
training/VarIsInitializedOp_9VarIsInitializedOptraining/Adam/m_16_1*
_output_shapes
: 
d
training/VarIsInitializedOp_10VarIsInitializedOptraining/Adam/vhat_5_1*
_output_shapes
: 
b
training/VarIsInitializedOp_11VarIsInitializedOptraining/Adam/v_15_1*
_output_shapes
: 
i
training/VarIsInitializedOp_12VarIsInitializedOpbatch_normalization_1/gamma*
_output_shapes
: 
o
training/VarIsInitializedOp_13VarIsInitializedOp!batch_normalization_2/moving_mean*
_output_shapes
: 
h
training/VarIsInitializedOp_14VarIsInitializedOpbatch_normalization_3/beta*
_output_shapes
: 
Z
training/VarIsInitializedOp_15VarIsInitializedOpdense_4/bias*
_output_shapes
: 
h
training/VarIsInitializedOp_16VarIsInitializedOpbatch_normalization_4/beta*
_output_shapes
: 
Y
training/VarIsInitializedOp_17VarIsInitializedOpAdam/beta_2*
_output_shapes
: 
e
training/VarIsInitializedOp_18VarIsInitializedOptraining/Adam/vhat_11_1*
_output_shapes
: 
e
training/VarIsInitializedOp_19VarIsInitializedOptraining/Adam/vhat_17_1*
_output_shapes
: 
b
training/VarIsInitializedOp_20VarIsInitializedOptraining/Adam/m_10_1*
_output_shapes
: 
a
training/VarIsInitializedOp_21VarIsInitializedOptraining/Adam/v_8_1*
_output_shapes
: 
b
training/VarIsInitializedOp_22VarIsInitializedOptraining/Adam/v_16_1*
_output_shapes
: 
b
training/VarIsInitializedOp_23VarIsInitializedOptraining/Adam/v_14_1*
_output_shapes
: 
d
training/VarIsInitializedOp_24VarIsInitializedOptraining/Adam/vhat_2_1*
_output_shapes
: 
s
training/VarIsInitializedOp_25VarIsInitializedOp%batch_normalization_1/moving_variance*
_output_shapes
: 
`
training/VarIsInitializedOp_26VarIsInitializedOpAdam/learning_rate*
_output_shapes
: 
a
training/VarIsInitializedOp_27VarIsInitializedOptraining/Adam/m_0_1*
_output_shapes
: 
d
training/VarIsInitializedOp_28VarIsInitializedOptraining/Adam/vhat_8_1*
_output_shapes
: 
e
training/VarIsInitializedOp_29VarIsInitializedOptraining/Adam/vhat_14_1*
_output_shapes
: 
i
training/VarIsInitializedOp_30VarIsInitializedOpbatch_normalization_2/gamma*
_output_shapes
: 
a
training/VarIsInitializedOp_31VarIsInitializedOptraining/Adam/m_4_1*
_output_shapes
: 
a
training/VarIsInitializedOp_32VarIsInitializedOptraining/Adam/m_9_1*
_output_shapes
: 
b
training/VarIsInitializedOp_33VarIsInitializedOptraining/Adam/m_13_1*
_output_shapes
: 
a
training/VarIsInitializedOp_34VarIsInitializedOptraining/Adam/v_4_1*
_output_shapes
: 
s
training/VarIsInitializedOp_35VarIsInitializedOp%batch_normalization_3/moving_variance*
_output_shapes
: 
a
training/VarIsInitializedOp_36VarIsInitializedOptraining/Adam/m_2_1*
_output_shapes
: 
a
training/VarIsInitializedOp_37VarIsInitializedOptraining/Adam/v_0_1*
_output_shapes
: 
b
training/VarIsInitializedOp_38VarIsInitializedOptraining/Adam/v_10_1*
_output_shapes
: 
\
training/VarIsInitializedOp_39VarIsInitializedOpdense_1/kernel*
_output_shapes
: 
\
training/VarIsInitializedOp_40VarIsInitializedOpdense_2/kernel*
_output_shapes
: 
a
training/VarIsInitializedOp_41VarIsInitializedOptraining/Adam/m_1_1*
_output_shapes
: 
a
training/VarIsInitializedOp_42VarIsInitializedOptraining/Adam/v_9_1*
_output_shapes
: 
a
training/VarIsInitializedOp_43VarIsInitializedOptraining/Adam/v_2_1*
_output_shapes
: 
b
training/VarIsInitializedOp_44VarIsInitializedOptraining/Adam/v_12_1*
_output_shapes
: 
d
training/VarIsInitializedOp_45VarIsInitializedOptraining/Adam/vhat_0_1*
_output_shapes
: 
d
training/VarIsInitializedOp_46VarIsInitializedOptraining/Adam/vhat_6_1*
_output_shapes
: 
b
training/VarIsInitializedOp_47VarIsInitializedOptraining/Adam/v_13_1*
_output_shapes
: 
i
training/VarIsInitializedOp_48VarIsInitializedOpbatch_normalization_3/gamma*
_output_shapes
: 
h
training/VarIsInitializedOp_49VarIsInitializedOpbatch_normalization_1/beta*
_output_shapes
: 
]
training/VarIsInitializedOp_50VarIsInitializedOpAdam/iterations*
_output_shapes
: 
S
training/VarIsInitializedOp_51VarIsInitializedOptotal*
_output_shapes
: 
b
training/VarIsInitializedOp_52VarIsInitializedOptraining/Adam/m_17_1*
_output_shapes
: 
d
training/VarIsInitializedOp_53VarIsInitializedOptraining/Adam/vhat_3_1*
_output_shapes
: 
h
training/VarIsInitializedOp_54VarIsInitializedOpbatch_normalization_2/beta*
_output_shapes
: 
s
training/VarIsInitializedOp_55VarIsInitializedOp%batch_normalization_4/moving_variance*
_output_shapes
: 
Z
training/VarIsInitializedOp_56VarIsInitializedOpdense_5/bias*
_output_shapes
: 
a
training/VarIsInitializedOp_57VarIsInitializedOptraining/Adam/v_1_1*
_output_shapes
: 
d
training/VarIsInitializedOp_58VarIsInitializedOptraining/Adam/vhat_9_1*
_output_shapes
: 
e
training/VarIsInitializedOp_59VarIsInitializedOptraining/Adam/vhat_15_1*
_output_shapes
: 
a
training/VarIsInitializedOp_60VarIsInitializedOptraining/Adam/m_7_1*
_output_shapes
: 
\
training/VarIsInitializedOp_61VarIsInitializedOpdense_4/kernel*
_output_shapes
: 
Y
training/VarIsInitializedOp_62VarIsInitializedOpAdam/beta_1*
_output_shapes
: 
b
training/VarIsInitializedOp_63VarIsInitializedOptraining/Adam/v_17_1*
_output_shapes
: 
a
training/VarIsInitializedOp_64VarIsInitializedOptraining/Adam/v_7_1*
_output_shapes
: 
o
training/VarIsInitializedOp_65VarIsInitializedOp!batch_normalization_4/moving_mean*
_output_shapes
: 
\
training/VarIsInitializedOp_66VarIsInitializedOpdense_3/kernel*
_output_shapes
: 
Z
training/VarIsInitializedOp_67VarIsInitializedOpdense_3/bias*
_output_shapes
: 
b
training/VarIsInitializedOp_68VarIsInitializedOptraining/Adam/m_11_1*
_output_shapes
: 
a
training/VarIsInitializedOp_69VarIsInitializedOptraining/Adam/v_3_1*
_output_shapes
: 
o
training/VarIsInitializedOp_70VarIsInitializedOp!batch_normalization_1/moving_mean*
_output_shapes
: 
i
training/VarIsInitializedOp_71VarIsInitializedOpbatch_normalization_4/gamma*
_output_shapes
: 
X
training/VarIsInitializedOp_72VarIsInitializedOp
Adam/decay*
_output_shapes
: 
a
training/VarIsInitializedOp_73VarIsInitializedOptraining/Adam/m_6_1*
_output_shapes
: 
d
training/VarIsInitializedOp_74VarIsInitializedOptraining/Adam/vhat_4_1*
_output_shapes
: 
b
training/VarIsInitializedOp_75VarIsInitializedOptraining/Adam/m_12_1*
_output_shapes
: 
e
training/VarIsInitializedOp_76VarIsInitializedOptraining/Adam/vhat_10_1*
_output_shapes
: 
e
training/VarIsInitializedOp_77VarIsInitializedOptraining/Adam/vhat_16_1*
_output_shapes
: 
a
training/VarIsInitializedOp_78VarIsInitializedOptraining/Adam/m_5_1*
_output_shapes
: 
Z
training/VarIsInitializedOp_79VarIsInitializedOpdense_2/bias*
_output_shapes
: 
d
training/VarIsInitializedOp_80VarIsInitializedOptraining/Adam/vhat_7_1*
_output_shapes
: 
d
training/VarIsInitializedOp_81VarIsInitializedOptraining/Adam/vhat_1_1*
_output_shapes
: 
b
training/VarIsInitializedOp_82VarIsInitializedOptraining/Adam/v_11_1*
_output_shapes
: 
e
training/VarIsInitializedOp_83VarIsInitializedOptraining/Adam/vhat_13_1*
_output_shapes
: 
a
training/VarIsInitializedOp_84VarIsInitializedOptraining/Adam/v_6_1*
_output_shapes
: 
s
training/VarIsInitializedOp_85VarIsInitializedOp%batch_normalization_2/moving_variance*
_output_shapes
: 
\
training/VarIsInitializedOp_86VarIsInitializedOpdense_5/kernel*
_output_shapes
: 
�
training/initNoOp^Adam/beta_1/Assign^Adam/beta_2/Assign^Adam/decay/Assign^Adam/iterations/Assign^Adam/learning_rate/Assign"^batch_normalization_1/beta/Assign#^batch_normalization_1/gamma/Assign)^batch_normalization_1/moving_mean/Assign-^batch_normalization_1/moving_variance/Assign"^batch_normalization_2/beta/Assign#^batch_normalization_2/gamma/Assign)^batch_normalization_2/moving_mean/Assign-^batch_normalization_2/moving_variance/Assign"^batch_normalization_3/beta/Assign#^batch_normalization_3/gamma/Assign)^batch_normalization_3/moving_mean/Assign-^batch_normalization_3/moving_variance/Assign"^batch_normalization_4/beta/Assign#^batch_normalization_4/gamma/Assign)^batch_normalization_4/moving_mean/Assign-^batch_normalization_4/moving_variance/Assign^count/Assign^dense_1/bias/Assign^dense_1/kernel/Assign^dense_2/bias/Assign^dense_2/kernel/Assign^dense_3/bias/Assign^dense_3/kernel/Assign^dense_4/bias/Assign^dense_4/kernel/Assign^dense_5/bias/Assign^dense_5/kernel/Assign^total/Assign^training/Adam/m_0_1/Assign^training/Adam/m_10_1/Assign^training/Adam/m_11_1/Assign^training/Adam/m_12_1/Assign^training/Adam/m_13_1/Assign^training/Adam/m_14_1/Assign^training/Adam/m_15_1/Assign^training/Adam/m_16_1/Assign^training/Adam/m_17_1/Assign^training/Adam/m_1_1/Assign^training/Adam/m_2_1/Assign^training/Adam/m_3_1/Assign^training/Adam/m_4_1/Assign^training/Adam/m_5_1/Assign^training/Adam/m_6_1/Assign^training/Adam/m_7_1/Assign^training/Adam/m_8_1/Assign^training/Adam/m_9_1/Assign^training/Adam/v_0_1/Assign^training/Adam/v_10_1/Assign^training/Adam/v_11_1/Assign^training/Adam/v_12_1/Assign^training/Adam/v_13_1/Assign^training/Adam/v_14_1/Assign^training/Adam/v_15_1/Assign^training/Adam/v_16_1/Assign^training/Adam/v_17_1/Assign^training/Adam/v_1_1/Assign^training/Adam/v_2_1/Assign^training/Adam/v_3_1/Assign^training/Adam/v_4_1/Assign^training/Adam/v_5_1/Assign^training/Adam/v_6_1/Assign^training/Adam/v_7_1/Assign^training/Adam/v_8_1/Assign^training/Adam/v_9_1/Assign^training/Adam/vhat_0_1/Assign^training/Adam/vhat_10_1/Assign^training/Adam/vhat_11_1/Assign^training/Adam/vhat_12_1/Assign^training/Adam/vhat_13_1/Assign^training/Adam/vhat_14_1/Assign^training/Adam/vhat_15_1/Assign^training/Adam/vhat_16_1/Assign^training/Adam/vhat_17_1/Assign^training/Adam/vhat_1_1/Assign^training/Adam/vhat_2_1/Assign^training/Adam/vhat_3_1/Assign^training/Adam/vhat_4_1/Assign^training/Adam/vhat_5_1/Assign^training/Adam/vhat_6_1/Assign^training/Adam/vhat_7_1/Assign^training/Adam/vhat_8_1/Assign^training/Adam/vhat_9_1/Assign
�
training/group_depsNoOp^Mean*^batch_normalization_1/AssignSubVariableOp,^batch_normalization_1/AssignSubVariableOp_1*^batch_normalization_2/AssignSubVariableOp,^batch_normalization_2/AssignSubVariableOp_1*^batch_normalization_3/AssignSubVariableOp,^batch_normalization_3/AssignSubVariableOp_1*^batch_normalization_4/AssignSubVariableOp,^batch_normalization_4/AssignSubVariableOp_1%^metrics/accuracy/AssignAddVariableOp'^metrics/accuracy/AssignAddVariableOp_1"^training/Adam/AssignAddVariableOp^training/Adam/AssignVariableOp!^training/Adam/AssignVariableOp_1"^training/Adam/AssignVariableOp_10"^training/Adam/AssignVariableOp_11"^training/Adam/AssignVariableOp_12"^training/Adam/AssignVariableOp_13"^training/Adam/AssignVariableOp_14"^training/Adam/AssignVariableOp_15"^training/Adam/AssignVariableOp_16"^training/Adam/AssignVariableOp_17"^training/Adam/AssignVariableOp_18"^training/Adam/AssignVariableOp_19!^training/Adam/AssignVariableOp_2"^training/Adam/AssignVariableOp_20"^training/Adam/AssignVariableOp_21"^training/Adam/AssignVariableOp_22"^training/Adam/AssignVariableOp_23"^training/Adam/AssignVariableOp_24"^training/Adam/AssignVariableOp_25"^training/Adam/AssignVariableOp_26"^training/Adam/AssignVariableOp_27"^training/Adam/AssignVariableOp_28"^training/Adam/AssignVariableOp_29!^training/Adam/AssignVariableOp_3"^training/Adam/AssignVariableOp_30"^training/Adam/AssignVariableOp_31"^training/Adam/AssignVariableOp_32"^training/Adam/AssignVariableOp_33"^training/Adam/AssignVariableOp_34"^training/Adam/AssignVariableOp_35"^training/Adam/AssignVariableOp_36"^training/Adam/AssignVariableOp_37"^training/Adam/AssignVariableOp_38"^training/Adam/AssignVariableOp_39!^training/Adam/AssignVariableOp_4"^training/Adam/AssignVariableOp_40"^training/Adam/AssignVariableOp_41"^training/Adam/AssignVariableOp_42"^training/Adam/AssignVariableOp_43"^training/Adam/AssignVariableOp_44"^training/Adam/AssignVariableOp_45"^training/Adam/AssignVariableOp_46"^training/Adam/AssignVariableOp_47"^training/Adam/AssignVariableOp_48"^training/Adam/AssignVariableOp_49!^training/Adam/AssignVariableOp_5"^training/Adam/AssignVariableOp_50"^training/Adam/AssignVariableOp_51"^training/Adam/AssignVariableOp_52"^training/Adam/AssignVariableOp_53!^training/Adam/AssignVariableOp_6!^training/Adam/AssignVariableOp_7!^training/Adam/AssignVariableOp_8!^training/Adam/AssignVariableOp_9
i

group_depsNoOp^Mean%^metrics/accuracy/AssignAddVariableOp'^metrics/accuracy/AssignAddVariableOp_1"������B     ۳��	�=}_��AJ�
�)�(
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	��
A
AddV2
x"T
y"T
z"T"
Ttype:
2	��
�
ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
E
AssignAddVariableOp
resource
value"dtype"
dtypetype�
E
AssignSubVariableOp
resource
value"dtype"
dtypetype�
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
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
S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
h
Equal
x"T
y"T
z
"
Ttype:
2	
"$
incompatible_shape_errorbool(�
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
9
FloorMod
x"T
y"T
z"T"
Ttype:

2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
?

LogSoftmax
logits"T

logsoftmax"T"
Ttype:
2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
8
Maximum
x"T
y"T
z"T"
Ttype:

2	
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
8
Minimum
x"T
y"T
z"T"
Ttype:

2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	�
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
6
Pow
x"T
y"T
z"T"
Ttype:

2	
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype�
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
.
Rsqrt
x"T
y"T"
Ttype:

2
;
	RsqrtGrad
y"T
dy"T
z"T"
Ttype:

2
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
O
Size

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
9
Softmax
logits"T
softmax"T"
Ttype:
2
j
SoftmaxCrossEntropyWithLogits
features"T
labels"T	
loss"T
backprop"T"
Ttype:
2
-
Sqrt
x"T
y"T"
Ttype:

2
1
Square
x"T
y"T"
Ttype:

2	
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	�
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
2
StopGradient

input"T
output"T"	
Ttype
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape�
9
VarIsInitializedOp
resource
is_initialized
�
O
VariableShape	
input
output"out_type"
out_typetype0:
2	�
&
	ZerosLike
x"T
y"T"	
Ttype*1.15.02unknown��
r
dense_1_inputPlaceholder*
shape:����������*
dtype0*(
_output_shapes
:����������
m
dense_1/random_uniform/shapeConst*
valueB"   �   *
dtype0*
_output_shapes
:
_
dense_1/random_uniform/minConst*
valueB
 *�\1�*
dtype0*
_output_shapes
: 
_
dense_1/random_uniform/maxConst*
valueB
 *�\1=*
dtype0*
_output_shapes
: 
�
$dense_1/random_uniform/RandomUniformRandomUniformdense_1/random_uniform/shape*
dtype0*
seed2���* 
_output_shapes
:
��*
seed���)*
T0
z
dense_1/random_uniform/subSubdense_1/random_uniform/maxdense_1/random_uniform/min*
_output_shapes
: *
T0
�
dense_1/random_uniform/mulMul$dense_1/random_uniform/RandomUniformdense_1/random_uniform/sub* 
_output_shapes
:
��*
T0
�
dense_1/random_uniformAdddense_1/random_uniform/muldense_1/random_uniform/min* 
_output_shapes
:
��*
T0
�
dense_1/kernelVarHandleOp*
shared_namedense_1/kernel*!
_class
loc:@dense_1/kernel*
	container *
shape:
��*
dtype0*
_output_shapes
: 
m
/dense_1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_1/kernel*
_output_shapes
: 
^
dense_1/kernel/AssignAssignVariableOpdense_1/kerneldense_1/random_uniform*
dtype0
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
dtype0* 
_output_shapes
:
��
\
dense_1/ConstConst*
valueB�*    *
dtype0*
_output_shapes	
:�
�
dense_1/biasVarHandleOp*
shape:�*
dtype0*
_output_shapes
: *
shared_namedense_1/bias*
_class
loc:@dense_1/bias*
	container 
i
-dense_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_1/bias*
_output_shapes
: 
Q
dense_1/bias/AssignAssignVariableOpdense_1/biasdense_1/Const*
dtype0
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
dtype0*
_output_shapes	
:�
n
dense_1/MatMul/ReadVariableOpReadVariableOpdense_1/kernel*
dtype0* 
_output_shapes
:
��
�
dense_1/MatMulMatMuldense_1_inputdense_1/MatMul/ReadVariableOp*
T0*
transpose_a( *(
_output_shapes
:����������*
transpose_b( 
h
dense_1/BiasAdd/ReadVariableOpReadVariableOpdense_1/bias*
dtype0*
_output_shapes	
:�
�
dense_1/BiasAddBiasAdddense_1/MatMuldense_1/BiasAdd/ReadVariableOp*
data_formatNHWC*(
_output_shapes
:����������*
T0
X
dense_1/ReluReludense_1/BiasAdd*
T0*(
_output_shapes
:����������
j
batch_normalization_1/ConstConst*
valueB�*  �?*
dtype0*
_output_shapes	
:�
�
batch_normalization_1/gammaVarHandleOp*
dtype0*
_output_shapes
: *,
shared_namebatch_normalization_1/gamma*.
_class$
" loc:@batch_normalization_1/gamma*
	container *
shape:�
�
<batch_normalization_1/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_1/gamma*
_output_shapes
: 
}
"batch_normalization_1/gamma/AssignAssignVariableOpbatch_normalization_1/gammabatch_normalization_1/Const*
dtype0
�
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
dtype0*
_output_shapes	
:�
l
batch_normalization_1/Const_1Const*
dtype0*
_output_shapes	
:�*
valueB�*    
�
batch_normalization_1/betaVarHandleOp*
shape:�*
dtype0*
_output_shapes
: *+
shared_namebatch_normalization_1/beta*-
_class#
!loc:@batch_normalization_1/beta*
	container 
�
;batch_normalization_1/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_1/beta*
_output_shapes
: 
}
!batch_normalization_1/beta/AssignAssignVariableOpbatch_normalization_1/betabatch_normalization_1/Const_1*
dtype0
�
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
dtype0*
_output_shapes	
:�
l
batch_normalization_1/Const_2Const*
valueB�*    *
dtype0*
_output_shapes	
:�
�
!batch_normalization_1/moving_meanVarHandleOp*4
_class*
(&loc:@batch_normalization_1/moving_mean*
	container *
shape:�*
dtype0*
_output_shapes
: *2
shared_name#!batch_normalization_1/moving_mean
�
Bbatch_normalization_1/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp!batch_normalization_1/moving_mean*
_output_shapes
: 
�
(batch_normalization_1/moving_mean/AssignAssignVariableOp!batch_normalization_1/moving_meanbatch_normalization_1/Const_2*
dtype0
�
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
dtype0*
_output_shapes	
:�
l
batch_normalization_1/Const_3Const*
valueB�*  �?*
dtype0*
_output_shapes	
:�
�
%batch_normalization_1/moving_varianceVarHandleOp*
dtype0*
_output_shapes
: *6
shared_name'%batch_normalization_1/moving_variance*8
_class.
,*loc:@batch_normalization_1/moving_variance*
	container *
shape:�
�
Fbatch_normalization_1/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp%batch_normalization_1/moving_variance*
_output_shapes
: 
�
,batch_normalization_1/moving_variance/AssignAssignVariableOp%batch_normalization_1/moving_variancebatch_normalization_1/Const_3*
dtype0
�
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
dtype0*
_output_shapes	
:�
~
4batch_normalization_1/moments/mean/reduction_indicesConst*
valueB: *
dtype0*
_output_shapes
:
�
"batch_normalization_1/moments/meanMeandense_1/Relu4batch_normalization_1/moments/mean/reduction_indices*

Tidx0*
	keep_dims(*
T0*
_output_shapes
:	�
�
*batch_normalization_1/moments/StopGradientStopGradient"batch_normalization_1/moments/mean*
T0*
_output_shapes
:	�
�
/batch_normalization_1/moments/SquaredDifferenceSquaredDifferencedense_1/Relu*batch_normalization_1/moments/StopGradient*(
_output_shapes
:����������*
T0
�
8batch_normalization_1/moments/variance/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB: 
�
&batch_normalization_1/moments/varianceMean/batch_normalization_1/moments/SquaredDifference8batch_normalization_1/moments/variance/reduction_indices*
T0*
_output_shapes
:	�*

Tidx0*
	keep_dims(
�
%batch_normalization_1/moments/SqueezeSqueeze"batch_normalization_1/moments/mean*
squeeze_dims
 *
T0*
_output_shapes	
:�
�
'batch_normalization_1/moments/Squeeze_1Squeeze&batch_normalization_1/moments/variance*
squeeze_dims
 *
T0*
_output_shapes	
:�
j
%batch_normalization_1/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *o�:
�
#batch_normalization_1/batchnorm/addAddV2'batch_normalization_1/moments/Squeeze_1%batch_normalization_1/batchnorm/add/y*
T0*
_output_shapes	
:�
y
%batch_normalization_1/batchnorm/RsqrtRsqrt#batch_normalization_1/batchnorm/add*
_output_shapes	
:�*
T0
�
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
dtype0*
_output_shapes	
:�
�
#batch_normalization_1/batchnorm/mulMul%batch_normalization_1/batchnorm/Rsqrt2batch_normalization_1/batchnorm/mul/ReadVariableOp*
_output_shapes	
:�*
T0
�
%batch_normalization_1/batchnorm/mul_1Muldense_1/Relu#batch_normalization_1/batchnorm/mul*
T0*(
_output_shapes
:����������
�
%batch_normalization_1/batchnorm/mul_2Mul%batch_normalization_1/moments/Squeeze#batch_normalization_1/batchnorm/mul*
T0*
_output_shapes	
:�
�
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
dtype0*
_output_shapes	
:�
�
#batch_normalization_1/batchnorm/subSub.batch_normalization_1/batchnorm/ReadVariableOp%batch_normalization_1/batchnorm/mul_2*
T0*
_output_shapes	
:�
�
%batch_normalization_1/batchnorm/add_1AddV2%batch_normalization_1/batchnorm/mul_1#batch_normalization_1/batchnorm/sub*
T0*(
_output_shapes
:����������
g
batch_normalization_1/ShapeShapedense_1/Relu*
T0*
out_type0*
_output_shapes
:
s
)batch_normalization_1/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
u
+batch_normalization_1/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
u
+batch_normalization_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
#batch_normalization_1/strided_sliceStridedSlicebatch_normalization_1/Shape)batch_normalization_1/strided_slice/stack+batch_normalization_1/strided_slice/stack_1+batch_normalization_1/strided_slice/stack_2*
end_mask *
_output_shapes
: *
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask 
�
!batch_normalization_1/Rank/packedPack#batch_normalization_1/strided_slice*
T0*

axis *
N*
_output_shapes
:
\
batch_normalization_1/RankConst*
value	B :*
dtype0*
_output_shapes
: 
c
!batch_normalization_1/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
c
!batch_normalization_1/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
�
batch_normalization_1/rangeRange!batch_normalization_1/range/startbatch_normalization_1/Rank!batch_normalization_1/range/delta*
_output_shapes
:*

Tidx0
�
 batch_normalization_1/Prod/inputPack#batch_normalization_1/strided_slice*
T0*

axis *
N*
_output_shapes
:
�
batch_normalization_1/ProdProd batch_normalization_1/Prod/inputbatch_normalization_1/range*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
~
batch_normalization_1/CastCastbatch_normalization_1/Prod*

SrcT0*
Truncate( *

DstT0*
_output_shapes
: 
`
batch_normalization_1/sub/yConst*
valueB
 *� �?*
dtype0*
_output_shapes
: 
z
batch_normalization_1/subSubbatch_normalization_1/Castbatch_normalization_1/sub/y*
T0*
_output_shapes
: 
�
batch_normalization_1/truedivRealDivbatch_normalization_1/Castbatch_normalization_1/sub*
T0*
_output_shapes
: 
�
batch_normalization_1/mulMul'batch_normalization_1/moments/Squeeze_1batch_normalization_1/truediv*
_output_shapes	
:�*
T0
�
batch_normalization_1/Const_4Const*4
_class*
(&loc:@batch_normalization_1/moving_mean*
valueB
 *
�#<*
dtype0*
_output_shapes
: 
�
$batch_normalization_1/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
dtype0*
_output_shapes	
:�
�
batch_normalization_1/sub_1Sub$batch_normalization_1/ReadVariableOp%batch_normalization_1/moments/Squeeze*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes	
:�
�
batch_normalization_1/mul_1Mulbatch_normalization_1/sub_1batch_normalization_1/Const_4*
_output_shapes	
:�*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean
�
)batch_normalization_1/AssignSubVariableOpAssignSubVariableOp!batch_normalization_1/moving_meanbatch_normalization_1/mul_1*4
_class*
(&loc:@batch_normalization_1/moving_mean*
dtype0
�
&batch_normalization_1/ReadVariableOp_1ReadVariableOp!batch_normalization_1/moving_mean*^batch_normalization_1/AssignSubVariableOp*4
_class*
(&loc:@batch_normalization_1/moving_mean*
dtype0*
_output_shapes	
:�
�
batch_normalization_1/Const_5Const*
dtype0*
_output_shapes
: *8
_class.
,*loc:@batch_normalization_1/moving_variance*
valueB
 *
�#<
�
&batch_normalization_1/ReadVariableOp_2ReadVariableOp%batch_normalization_1/moving_variance*
dtype0*
_output_shapes	
:�
�
batch_normalization_1/sub_2Sub&batch_normalization_1/ReadVariableOp_2batch_normalization_1/mul*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes	
:�
�
batch_normalization_1/mul_2Mulbatch_normalization_1/sub_2batch_normalization_1/Const_5*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes	
:�
�
+batch_normalization_1/AssignSubVariableOp_1AssignSubVariableOp%batch_normalization_1/moving_variancebatch_normalization_1/mul_2*
dtype0*8
_class.
,*loc:@batch_normalization_1/moving_variance
�
&batch_normalization_1/ReadVariableOp_3ReadVariableOp%batch_normalization_1/moving_variance,^batch_normalization_1/AssignSubVariableOp_1*
dtype0*
_output_shapes	
:�*8
_class.
,*loc:@batch_normalization_1/moving_variance
\
keras_learning_phase/inputConst*
dtype0
*
_output_shapes
: *
value	B
 Z 
|
keras_learning_phasePlaceholderWithDefaultkeras_learning_phase/input*
dtype0
*
_output_shapes
: *
shape: 
z
!batch_normalization_1/cond/SwitchSwitchkeras_learning_phasekeras_learning_phase*
_output_shapes
: : *
T0

u
#batch_normalization_1/cond/switch_tIdentity#batch_normalization_1/cond/Switch:1*
T0
*
_output_shapes
: 
s
#batch_normalization_1/cond/switch_fIdentity!batch_normalization_1/cond/Switch*
T0
*
_output_shapes
: 
e
"batch_normalization_1/cond/pred_idIdentitykeras_learning_phase*
_output_shapes
: *
T0

�
#batch_normalization_1/cond/Switch_1Switch%batch_normalization_1/batchnorm/add_1"batch_normalization_1/cond/pred_id*<
_output_shapes*
(:����������:����������*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1
�
3batch_normalization_1/cond/batchnorm/ReadVariableOpReadVariableOp:batch_normalization_1/cond/batchnorm/ReadVariableOp/Switch*
dtype0*
_output_shapes	
:�
�
:batch_normalization_1/cond/batchnorm/ReadVariableOp/SwitchSwitch%batch_normalization_1/moving_variance"batch_normalization_1/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes
: : 
�
*batch_normalization_1/cond/batchnorm/add/yConst$^batch_normalization_1/cond/switch_f*
dtype0*
_output_shapes
: *
valueB
 *o�:
�
(batch_normalization_1/cond/batchnorm/addAddV23batch_normalization_1/cond/batchnorm/ReadVariableOp*batch_normalization_1/cond/batchnorm/add/y*
T0*
_output_shapes	
:�
�
*batch_normalization_1/cond/batchnorm/RsqrtRsqrt(batch_normalization_1/cond/batchnorm/add*
T0*
_output_shapes	
:�
�
7batch_normalization_1/cond/batchnorm/mul/ReadVariableOpReadVariableOp>batch_normalization_1/cond/batchnorm/mul/ReadVariableOp/Switch*
dtype0*
_output_shapes	
:�
�
>batch_normalization_1/cond/batchnorm/mul/ReadVariableOp/SwitchSwitchbatch_normalization_1/gamma"batch_normalization_1/cond/pred_id*
_output_shapes
: : *
T0*.
_class$
" loc:@batch_normalization_1/gamma
�
(batch_normalization_1/cond/batchnorm/mulMul*batch_normalization_1/cond/batchnorm/Rsqrt7batch_normalization_1/cond/batchnorm/mul/ReadVariableOp*
_output_shapes	
:�*
T0
�
*batch_normalization_1/cond/batchnorm/mul_1Mul1batch_normalization_1/cond/batchnorm/mul_1/Switch(batch_normalization_1/cond/batchnorm/mul*(
_output_shapes
:����������*
T0
�
1batch_normalization_1/cond/batchnorm/mul_1/SwitchSwitchdense_1/Relu"batch_normalization_1/cond/pred_id*
T0*
_class
loc:@dense_1/Relu*<
_output_shapes*
(:����������:����������
�
5batch_normalization_1/cond/batchnorm/ReadVariableOp_1ReadVariableOp<batch_normalization_1/cond/batchnorm/ReadVariableOp_1/Switch*
dtype0*
_output_shapes	
:�
�
<batch_normalization_1/cond/batchnorm/ReadVariableOp_1/SwitchSwitch!batch_normalization_1/moving_mean"batch_normalization_1/cond/pred_id*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes
: : 
�
*batch_normalization_1/cond/batchnorm/mul_2Mul5batch_normalization_1/cond/batchnorm/ReadVariableOp_1(batch_normalization_1/cond/batchnorm/mul*
_output_shapes	
:�*
T0
�
5batch_normalization_1/cond/batchnorm/ReadVariableOp_2ReadVariableOp<batch_normalization_1/cond/batchnorm/ReadVariableOp_2/Switch*
dtype0*
_output_shapes	
:�
�
<batch_normalization_1/cond/batchnorm/ReadVariableOp_2/SwitchSwitchbatch_normalization_1/beta"batch_normalization_1/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_1/beta*
_output_shapes
: : 
�
(batch_normalization_1/cond/batchnorm/subSub5batch_normalization_1/cond/batchnorm/ReadVariableOp_2*batch_normalization_1/cond/batchnorm/mul_2*
T0*
_output_shapes	
:�
�
*batch_normalization_1/cond/batchnorm/add_1AddV2*batch_normalization_1/cond/batchnorm/mul_1(batch_normalization_1/cond/batchnorm/sub*
T0*(
_output_shapes
:����������
�
 batch_normalization_1/cond/MergeMerge*batch_normalization_1/cond/batchnorm/add_1%batch_normalization_1/cond/Switch_1:1*
T0*
N**
_output_shapes
:����������: 
m
dense_2/random_uniform/shapeConst*
valueB"�   �   *
dtype0*
_output_shapes
:
_
dense_2/random_uniform/minConst*
valueB
 *q��*
dtype0*
_output_shapes
: 
_
dense_2/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *q�>
�
$dense_2/random_uniform/RandomUniformRandomUniformdense_2/random_uniform/shape*
dtype0*
seed2ئ�* 
_output_shapes
:
��*
seed���)*
T0
z
dense_2/random_uniform/subSubdense_2/random_uniform/maxdense_2/random_uniform/min*
T0*
_output_shapes
: 
�
dense_2/random_uniform/mulMul$dense_2/random_uniform/RandomUniformdense_2/random_uniform/sub* 
_output_shapes
:
��*
T0
�
dense_2/random_uniformAdddense_2/random_uniform/muldense_2/random_uniform/min*
T0* 
_output_shapes
:
��
�
dense_2/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shared_namedense_2/kernel*!
_class
loc:@dense_2/kernel*
	container *
shape:
��
m
/dense_2/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_2/kernel*
_output_shapes
: 
^
dense_2/kernel/AssignAssignVariableOpdense_2/kerneldense_2/random_uniform*
dtype0
s
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
dtype0* 
_output_shapes
:
��
\
dense_2/ConstConst*
valueB�*    *
dtype0*
_output_shapes	
:�
�
dense_2/biasVarHandleOp*
_class
loc:@dense_2/bias*
	container *
shape:�*
dtype0*
_output_shapes
: *
shared_namedense_2/bias
i
-dense_2/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_2/bias*
_output_shapes
: 
Q
dense_2/bias/AssignAssignVariableOpdense_2/biasdense_2/Const*
dtype0
j
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
dtype0*
_output_shapes	
:�
n
dense_2/MatMul/ReadVariableOpReadVariableOpdense_2/kernel*
dtype0* 
_output_shapes
:
��
�
dense_2/MatMulMatMul batch_normalization_1/cond/Mergedense_2/MatMul/ReadVariableOp*
transpose_a( *(
_output_shapes
:����������*
transpose_b( *
T0
h
dense_2/BiasAdd/ReadVariableOpReadVariableOpdense_2/bias*
dtype0*
_output_shapes	
:�
�
dense_2/BiasAddBiasAdddense_2/MatMuldense_2/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*(
_output_shapes
:����������
X
dense_2/ReluReludense_2/BiasAdd*
T0*(
_output_shapes
:����������
j
batch_normalization_2/ConstConst*
valueB�*  �?*
dtype0*
_output_shapes	
:�
�
batch_normalization_2/gammaVarHandleOp*,
shared_namebatch_normalization_2/gamma*.
_class$
" loc:@batch_normalization_2/gamma*
	container *
shape:�*
dtype0*
_output_shapes
: 
�
<batch_normalization_2/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_2/gamma*
_output_shapes
: 
}
"batch_normalization_2/gamma/AssignAssignVariableOpbatch_normalization_2/gammabatch_normalization_2/Const*
dtype0
�
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
dtype0*
_output_shapes	
:�
l
batch_normalization_2/Const_1Const*
valueB�*    *
dtype0*
_output_shapes	
:�
�
batch_normalization_2/betaVarHandleOp*+
shared_namebatch_normalization_2/beta*-
_class#
!loc:@batch_normalization_2/beta*
	container *
shape:�*
dtype0*
_output_shapes
: 
�
;batch_normalization_2/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_2/beta*
_output_shapes
: 
}
!batch_normalization_2/beta/AssignAssignVariableOpbatch_normalization_2/betabatch_normalization_2/Const_1*
dtype0
�
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
dtype0*
_output_shapes	
:�
l
batch_normalization_2/Const_2Const*
valueB�*    *
dtype0*
_output_shapes	
:�
�
!batch_normalization_2/moving_meanVarHandleOp*2
shared_name#!batch_normalization_2/moving_mean*4
_class*
(&loc:@batch_normalization_2/moving_mean*
	container *
shape:�*
dtype0*
_output_shapes
: 
�
Bbatch_normalization_2/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp!batch_normalization_2/moving_mean*
_output_shapes
: 
�
(batch_normalization_2/moving_mean/AssignAssignVariableOp!batch_normalization_2/moving_meanbatch_normalization_2/Const_2*
dtype0
�
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
dtype0*
_output_shapes	
:�
l
batch_normalization_2/Const_3Const*
valueB�*  �?*
dtype0*
_output_shapes	
:�
�
%batch_normalization_2/moving_varianceVarHandleOp*
	container *
shape:�*
dtype0*
_output_shapes
: *6
shared_name'%batch_normalization_2/moving_variance*8
_class.
,*loc:@batch_normalization_2/moving_variance
�
Fbatch_normalization_2/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp%batch_normalization_2/moving_variance*
_output_shapes
: 
�
,batch_normalization_2/moving_variance/AssignAssignVariableOp%batch_normalization_2/moving_variancebatch_normalization_2/Const_3*
dtype0
�
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
dtype0*
_output_shapes	
:�
~
4batch_normalization_2/moments/mean/reduction_indicesConst*
valueB: *
dtype0*
_output_shapes
:
�
"batch_normalization_2/moments/meanMeandense_2/Relu4batch_normalization_2/moments/mean/reduction_indices*
T0*
_output_shapes
:	�*

Tidx0*
	keep_dims(
�
*batch_normalization_2/moments/StopGradientStopGradient"batch_normalization_2/moments/mean*
T0*
_output_shapes
:	�
�
/batch_normalization_2/moments/SquaredDifferenceSquaredDifferencedense_2/Relu*batch_normalization_2/moments/StopGradient*
T0*(
_output_shapes
:����������
�
8batch_normalization_2/moments/variance/reduction_indicesConst*
valueB: *
dtype0*
_output_shapes
:
�
&batch_normalization_2/moments/varianceMean/batch_normalization_2/moments/SquaredDifference8batch_normalization_2/moments/variance/reduction_indices*
T0*
_output_shapes
:	�*

Tidx0*
	keep_dims(
�
%batch_normalization_2/moments/SqueezeSqueeze"batch_normalization_2/moments/mean*
_output_shapes	
:�*
squeeze_dims
 *
T0
�
'batch_normalization_2/moments/Squeeze_1Squeeze&batch_normalization_2/moments/variance*
squeeze_dims
 *
T0*
_output_shapes	
:�
j
%batch_normalization_2/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
#batch_normalization_2/batchnorm/addAddV2'batch_normalization_2/moments/Squeeze_1%batch_normalization_2/batchnorm/add/y*
T0*
_output_shapes	
:�
y
%batch_normalization_2/batchnorm/RsqrtRsqrt#batch_normalization_2/batchnorm/add*
T0*
_output_shapes	
:�
�
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
dtype0*
_output_shapes	
:�
�
#batch_normalization_2/batchnorm/mulMul%batch_normalization_2/batchnorm/Rsqrt2batch_normalization_2/batchnorm/mul/ReadVariableOp*
T0*
_output_shapes	
:�
�
%batch_normalization_2/batchnorm/mul_1Muldense_2/Relu#batch_normalization_2/batchnorm/mul*(
_output_shapes
:����������*
T0
�
%batch_normalization_2/batchnorm/mul_2Mul%batch_normalization_2/moments/Squeeze#batch_normalization_2/batchnorm/mul*
T0*
_output_shapes	
:�
�
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
dtype0*
_output_shapes	
:�
�
#batch_normalization_2/batchnorm/subSub.batch_normalization_2/batchnorm/ReadVariableOp%batch_normalization_2/batchnorm/mul_2*
T0*
_output_shapes	
:�
�
%batch_normalization_2/batchnorm/add_1AddV2%batch_normalization_2/batchnorm/mul_1#batch_normalization_2/batchnorm/sub*
T0*(
_output_shapes
:����������
g
batch_normalization_2/ShapeShapedense_2/Relu*
T0*
out_type0*
_output_shapes
:
s
)batch_normalization_2/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
u
+batch_normalization_2/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
u
+batch_normalization_2/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
#batch_normalization_2/strided_sliceStridedSlicebatch_normalization_2/Shape)batch_normalization_2/strided_slice/stack+batch_normalization_2/strided_slice/stack_1+batch_normalization_2/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
�
!batch_normalization_2/Rank/packedPack#batch_normalization_2/strided_slice*
N*
_output_shapes
:*
T0*

axis 
\
batch_normalization_2/RankConst*
value	B :*
dtype0*
_output_shapes
: 
c
!batch_normalization_2/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
c
!batch_normalization_2/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
batch_normalization_2/rangeRange!batch_normalization_2/range/startbatch_normalization_2/Rank!batch_normalization_2/range/delta*
_output_shapes
:*

Tidx0
�
 batch_normalization_2/Prod/inputPack#batch_normalization_2/strided_slice*
T0*

axis *
N*
_output_shapes
:
�
batch_normalization_2/ProdProd batch_normalization_2/Prod/inputbatch_normalization_2/range*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
~
batch_normalization_2/CastCastbatch_normalization_2/Prod*

SrcT0*
Truncate( *

DstT0*
_output_shapes
: 
`
batch_normalization_2/sub/yConst*
valueB
 *� �?*
dtype0*
_output_shapes
: 
z
batch_normalization_2/subSubbatch_normalization_2/Castbatch_normalization_2/sub/y*
T0*
_output_shapes
: 
�
batch_normalization_2/truedivRealDivbatch_normalization_2/Castbatch_normalization_2/sub*
T0*
_output_shapes
: 
�
batch_normalization_2/mulMul'batch_normalization_2/moments/Squeeze_1batch_normalization_2/truediv*
T0*
_output_shapes	
:�
�
batch_normalization_2/Const_4Const*4
_class*
(&loc:@batch_normalization_2/moving_mean*
valueB
 *
�#<*
dtype0*
_output_shapes
: 
�
$batch_normalization_2/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
dtype0*
_output_shapes	
:�
�
batch_normalization_2/sub_1Sub$batch_normalization_2/ReadVariableOp%batch_normalization_2/moments/Squeeze*
_output_shapes	
:�*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean
�
batch_normalization_2/mul_1Mulbatch_normalization_2/sub_1batch_normalization_2/Const_4*
_output_shapes	
:�*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean
�
)batch_normalization_2/AssignSubVariableOpAssignSubVariableOp!batch_normalization_2/moving_meanbatch_normalization_2/mul_1*
dtype0*4
_class*
(&loc:@batch_normalization_2/moving_mean
�
&batch_normalization_2/ReadVariableOp_1ReadVariableOp!batch_normalization_2/moving_mean*^batch_normalization_2/AssignSubVariableOp*4
_class*
(&loc:@batch_normalization_2/moving_mean*
dtype0*
_output_shapes	
:�
�
batch_normalization_2/Const_5Const*
dtype0*
_output_shapes
: *8
_class.
,*loc:@batch_normalization_2/moving_variance*
valueB
 *
�#<
�
&batch_normalization_2/ReadVariableOp_2ReadVariableOp%batch_normalization_2/moving_variance*
dtype0*
_output_shapes	
:�
�
batch_normalization_2/sub_2Sub&batch_normalization_2/ReadVariableOp_2batch_normalization_2/mul*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes	
:�
�
batch_normalization_2/mul_2Mulbatch_normalization_2/sub_2batch_normalization_2/Const_5*
_output_shapes	
:�*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance
�
+batch_normalization_2/AssignSubVariableOp_1AssignSubVariableOp%batch_normalization_2/moving_variancebatch_normalization_2/mul_2*
dtype0*8
_class.
,*loc:@batch_normalization_2/moving_variance
�
&batch_normalization_2/ReadVariableOp_3ReadVariableOp%batch_normalization_2/moving_variance,^batch_normalization_2/AssignSubVariableOp_1*
dtype0*
_output_shapes	
:�*8
_class.
,*loc:@batch_normalization_2/moving_variance
z
!batch_normalization_2/cond/SwitchSwitchkeras_learning_phasekeras_learning_phase*
_output_shapes
: : *
T0

u
#batch_normalization_2/cond/switch_tIdentity#batch_normalization_2/cond/Switch:1*
_output_shapes
: *
T0

s
#batch_normalization_2/cond/switch_fIdentity!batch_normalization_2/cond/Switch*
T0
*
_output_shapes
: 
e
"batch_normalization_2/cond/pred_idIdentitykeras_learning_phase*
T0
*
_output_shapes
: 
�
#batch_normalization_2/cond/Switch_1Switch%batch_normalization_2/batchnorm/add_1"batch_normalization_2/cond/pred_id*<
_output_shapes*
(:����������:����������*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/add_1
�
3batch_normalization_2/cond/batchnorm/ReadVariableOpReadVariableOp:batch_normalization_2/cond/batchnorm/ReadVariableOp/Switch*
dtype0*
_output_shapes	
:�
�
:batch_normalization_2/cond/batchnorm/ReadVariableOp/SwitchSwitch%batch_normalization_2/moving_variance"batch_normalization_2/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes
: : 
�
*batch_normalization_2/cond/batchnorm/add/yConst$^batch_normalization_2/cond/switch_f*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
(batch_normalization_2/cond/batchnorm/addAddV23batch_normalization_2/cond/batchnorm/ReadVariableOp*batch_normalization_2/cond/batchnorm/add/y*
T0*
_output_shapes	
:�
�
*batch_normalization_2/cond/batchnorm/RsqrtRsqrt(batch_normalization_2/cond/batchnorm/add*
T0*
_output_shapes	
:�
�
7batch_normalization_2/cond/batchnorm/mul/ReadVariableOpReadVariableOp>batch_normalization_2/cond/batchnorm/mul/ReadVariableOp/Switch*
dtype0*
_output_shapes	
:�
�
>batch_normalization_2/cond/batchnorm/mul/ReadVariableOp/SwitchSwitchbatch_normalization_2/gamma"batch_normalization_2/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_2/gamma*
_output_shapes
: : 
�
(batch_normalization_2/cond/batchnorm/mulMul*batch_normalization_2/cond/batchnorm/Rsqrt7batch_normalization_2/cond/batchnorm/mul/ReadVariableOp*
T0*
_output_shapes	
:�
�
*batch_normalization_2/cond/batchnorm/mul_1Mul1batch_normalization_2/cond/batchnorm/mul_1/Switch(batch_normalization_2/cond/batchnorm/mul*(
_output_shapes
:����������*
T0
�
1batch_normalization_2/cond/batchnorm/mul_1/SwitchSwitchdense_2/Relu"batch_normalization_2/cond/pred_id*
T0*
_class
loc:@dense_2/Relu*<
_output_shapes*
(:����������:����������
�
5batch_normalization_2/cond/batchnorm/ReadVariableOp_1ReadVariableOp<batch_normalization_2/cond/batchnorm/ReadVariableOp_1/Switch*
dtype0*
_output_shapes	
:�
�
<batch_normalization_2/cond/batchnorm/ReadVariableOp_1/SwitchSwitch!batch_normalization_2/moving_mean"batch_normalization_2/cond/pred_id*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean*
_output_shapes
: : 
�
*batch_normalization_2/cond/batchnorm/mul_2Mul5batch_normalization_2/cond/batchnorm/ReadVariableOp_1(batch_normalization_2/cond/batchnorm/mul*
_output_shapes	
:�*
T0
�
5batch_normalization_2/cond/batchnorm/ReadVariableOp_2ReadVariableOp<batch_normalization_2/cond/batchnorm/ReadVariableOp_2/Switch*
dtype0*
_output_shapes	
:�
�
<batch_normalization_2/cond/batchnorm/ReadVariableOp_2/SwitchSwitchbatch_normalization_2/beta"batch_normalization_2/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_2/beta*
_output_shapes
: : 
�
(batch_normalization_2/cond/batchnorm/subSub5batch_normalization_2/cond/batchnorm/ReadVariableOp_2*batch_normalization_2/cond/batchnorm/mul_2*
T0*
_output_shapes	
:�
�
*batch_normalization_2/cond/batchnorm/add_1AddV2*batch_normalization_2/cond/batchnorm/mul_1(batch_normalization_2/cond/batchnorm/sub*(
_output_shapes
:����������*
T0
�
 batch_normalization_2/cond/MergeMerge*batch_normalization_2/cond/batchnorm/add_1%batch_normalization_2/cond/Switch_1:1*
N**
_output_shapes
:����������: *
T0
n
dropout_1/cond/SwitchSwitchkeras_learning_phasekeras_learning_phase*
T0
*
_output_shapes
: : 
]
dropout_1/cond/switch_tIdentitydropout_1/cond/Switch:1*
T0
*
_output_shapes
: 
[
dropout_1/cond/switch_fIdentitydropout_1/cond/Switch*
T0
*
_output_shapes
: 
Y
dropout_1/cond/pred_idIdentitykeras_learning_phase*
T0
*
_output_shapes
: 
z
dropout_1/cond/dropout/rateConst^dropout_1/cond/switch_t*
valueB
 *   ?*
dtype0*
_output_shapes
: 
�
dropout_1/cond/dropout/ShapeShape%dropout_1/cond/dropout/Shape/Switch:1*
_output_shapes
:*
T0*
out_type0
�
#dropout_1/cond/dropout/Shape/SwitchSwitch batch_normalization_2/cond/Mergedropout_1/cond/pred_id*
T0*3
_class)
'%loc:@batch_normalization_2/cond/Merge*<
_output_shapes*
(:����������:����������
�
)dropout_1/cond/dropout/random_uniform/minConst^dropout_1/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
�
)dropout_1/cond/dropout/random_uniform/maxConst^dropout_1/cond/switch_t*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
3dropout_1/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_1/cond/dropout/Shape*
dtype0*
seed2���*(
_output_shapes
:����������*
seed���)*
T0
�
)dropout_1/cond/dropout/random_uniform/subSub)dropout_1/cond/dropout/random_uniform/max)dropout_1/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
�
)dropout_1/cond/dropout/random_uniform/mulMul3dropout_1/cond/dropout/random_uniform/RandomUniform)dropout_1/cond/dropout/random_uniform/sub*
T0*(
_output_shapes
:����������
�
%dropout_1/cond/dropout/random_uniformAdd)dropout_1/cond/dropout/random_uniform/mul)dropout_1/cond/dropout/random_uniform/min*
T0*(
_output_shapes
:����������
{
dropout_1/cond/dropout/sub/xConst^dropout_1/cond/switch_t*
valueB
 *  �?*
dtype0*
_output_shapes
: 
}
dropout_1/cond/dropout/subSubdropout_1/cond/dropout/sub/xdropout_1/cond/dropout/rate*
_output_shapes
: *
T0

 dropout_1/cond/dropout/truediv/xConst^dropout_1/cond/switch_t*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
dropout_1/cond/dropout/truedivRealDiv dropout_1/cond/dropout/truediv/xdropout_1/cond/dropout/sub*
T0*
_output_shapes
: 
�
#dropout_1/cond/dropout/GreaterEqualGreaterEqual%dropout_1/cond/dropout/random_uniformdropout_1/cond/dropout/rate*
T0*(
_output_shapes
:����������
�
dropout_1/cond/dropout/mulMul%dropout_1/cond/dropout/Shape/Switch:1dropout_1/cond/dropout/truediv*
T0*(
_output_shapes
:����������
�
dropout_1/cond/dropout/CastCast#dropout_1/cond/dropout/GreaterEqual*

SrcT0
*
Truncate( *

DstT0*(
_output_shapes
:����������
�
dropout_1/cond/dropout/mul_1Muldropout_1/cond/dropout/muldropout_1/cond/dropout/Cast*
T0*(
_output_shapes
:����������
�
dropout_1/cond/Switch_1Switch batch_normalization_2/cond/Mergedropout_1/cond/pred_id*
T0*3
_class)
'%loc:@batch_normalization_2/cond/Merge*<
_output_shapes*
(:����������:����������
�
dropout_1/cond/MergeMergedropout_1/cond/Switch_1dropout_1/cond/dropout/mul_1*
T0*
N**
_output_shapes
:����������: 
m
dense_3/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"�   �   
_
dense_3/random_uniform/minConst*
valueB
 *q��*
dtype0*
_output_shapes
: 
_
dense_3/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *q�>
�
$dense_3/random_uniform/RandomUniformRandomUniformdense_3/random_uniform/shape*
dtype0*
seed2�֌* 
_output_shapes
:
��*
seed���)*
T0
z
dense_3/random_uniform/subSubdense_3/random_uniform/maxdense_3/random_uniform/min*
T0*
_output_shapes
: 
�
dense_3/random_uniform/mulMul$dense_3/random_uniform/RandomUniformdense_3/random_uniform/sub*
T0* 
_output_shapes
:
��
�
dense_3/random_uniformAdddense_3/random_uniform/muldense_3/random_uniform/min*
T0* 
_output_shapes
:
��
�
dense_3/kernelVarHandleOp*
shared_namedense_3/kernel*!
_class
loc:@dense_3/kernel*
	container *
shape:
��*
dtype0*
_output_shapes
: 
m
/dense_3/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_3/kernel*
_output_shapes
: 
^
dense_3/kernel/AssignAssignVariableOpdense_3/kerneldense_3/random_uniform*
dtype0
s
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
dtype0* 
_output_shapes
:
��
\
dense_3/ConstConst*
dtype0*
_output_shapes	
:�*
valueB�*    
�
dense_3/biasVarHandleOp*
dtype0*
_output_shapes
: *
shared_namedense_3/bias*
_class
loc:@dense_3/bias*
	container *
shape:�
i
-dense_3/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_3/bias*
_output_shapes
: 
Q
dense_3/bias/AssignAssignVariableOpdense_3/biasdense_3/Const*
dtype0
j
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
dtype0*
_output_shapes	
:�
n
dense_3/MatMul/ReadVariableOpReadVariableOpdense_3/kernel*
dtype0* 
_output_shapes
:
��
�
dense_3/MatMulMatMuldropout_1/cond/Mergedense_3/MatMul/ReadVariableOp*
transpose_b( *
T0*
transpose_a( *(
_output_shapes
:����������
h
dense_3/BiasAdd/ReadVariableOpReadVariableOpdense_3/bias*
dtype0*
_output_shapes	
:�
�
dense_3/BiasAddBiasAdddense_3/MatMuldense_3/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*(
_output_shapes
:����������
X
dense_3/ReluReludense_3/BiasAdd*
T0*(
_output_shapes
:����������
j
batch_normalization_3/ConstConst*
valueB�*  �?*
dtype0*
_output_shapes	
:�
�
batch_normalization_3/gammaVarHandleOp*,
shared_namebatch_normalization_3/gamma*.
_class$
" loc:@batch_normalization_3/gamma*
	container *
shape:�*
dtype0*
_output_shapes
: 
�
<batch_normalization_3/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_3/gamma*
_output_shapes
: 
}
"batch_normalization_3/gamma/AssignAssignVariableOpbatch_normalization_3/gammabatch_normalization_3/Const*
dtype0
�
/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
dtype0*
_output_shapes	
:�
l
batch_normalization_3/Const_1Const*
valueB�*    *
dtype0*
_output_shapes	
:�
�
batch_normalization_3/betaVarHandleOp*-
_class#
!loc:@batch_normalization_3/beta*
	container *
shape:�*
dtype0*
_output_shapes
: *+
shared_namebatch_normalization_3/beta
�
;batch_normalization_3/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_3/beta*
_output_shapes
: 
}
!batch_normalization_3/beta/AssignAssignVariableOpbatch_normalization_3/betabatch_normalization_3/Const_1*
dtype0
�
.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
dtype0*
_output_shapes	
:�
l
batch_normalization_3/Const_2Const*
valueB�*    *
dtype0*
_output_shapes	
:�
�
!batch_normalization_3/moving_meanVarHandleOp*4
_class*
(&loc:@batch_normalization_3/moving_mean*
	container *
shape:�*
dtype0*
_output_shapes
: *2
shared_name#!batch_normalization_3/moving_mean
�
Bbatch_normalization_3/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp!batch_normalization_3/moving_mean*
_output_shapes
: 
�
(batch_normalization_3/moving_mean/AssignAssignVariableOp!batch_normalization_3/moving_meanbatch_normalization_3/Const_2*
dtype0
�
5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
dtype0*
_output_shapes	
:�
l
batch_normalization_3/Const_3Const*
dtype0*
_output_shapes	
:�*
valueB�*  �?
�
%batch_normalization_3/moving_varianceVarHandleOp*6
shared_name'%batch_normalization_3/moving_variance*8
_class.
,*loc:@batch_normalization_3/moving_variance*
	container *
shape:�*
dtype0*
_output_shapes
: 
�
Fbatch_normalization_3/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp%batch_normalization_3/moving_variance*
_output_shapes
: 
�
,batch_normalization_3/moving_variance/AssignAssignVariableOp%batch_normalization_3/moving_variancebatch_normalization_3/Const_3*
dtype0
�
9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
dtype0*
_output_shapes	
:�
~
4batch_normalization_3/moments/mean/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB: 
�
"batch_normalization_3/moments/meanMeandense_3/Relu4batch_normalization_3/moments/mean/reduction_indices*

Tidx0*
	keep_dims(*
T0*
_output_shapes
:	�
�
*batch_normalization_3/moments/StopGradientStopGradient"batch_normalization_3/moments/mean*
T0*
_output_shapes
:	�
�
/batch_normalization_3/moments/SquaredDifferenceSquaredDifferencedense_3/Relu*batch_normalization_3/moments/StopGradient*
T0*(
_output_shapes
:����������
�
8batch_normalization_3/moments/variance/reduction_indicesConst*
valueB: *
dtype0*
_output_shapes
:
�
&batch_normalization_3/moments/varianceMean/batch_normalization_3/moments/SquaredDifference8batch_normalization_3/moments/variance/reduction_indices*
_output_shapes
:	�*

Tidx0*
	keep_dims(*
T0
�
%batch_normalization_3/moments/SqueezeSqueeze"batch_normalization_3/moments/mean*
squeeze_dims
 *
T0*
_output_shapes	
:�
�
'batch_normalization_3/moments/Squeeze_1Squeeze&batch_normalization_3/moments/variance*
_output_shapes	
:�*
squeeze_dims
 *
T0
j
%batch_normalization_3/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
#batch_normalization_3/batchnorm/addAddV2'batch_normalization_3/moments/Squeeze_1%batch_normalization_3/batchnorm/add/y*
T0*
_output_shapes	
:�
y
%batch_normalization_3/batchnorm/RsqrtRsqrt#batch_normalization_3/batchnorm/add*
T0*
_output_shapes	
:�
�
2batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
dtype0*
_output_shapes	
:�
�
#batch_normalization_3/batchnorm/mulMul%batch_normalization_3/batchnorm/Rsqrt2batch_normalization_3/batchnorm/mul/ReadVariableOp*
T0*
_output_shapes	
:�
�
%batch_normalization_3/batchnorm/mul_1Muldense_3/Relu#batch_normalization_3/batchnorm/mul*
T0*(
_output_shapes
:����������
�
%batch_normalization_3/batchnorm/mul_2Mul%batch_normalization_3/moments/Squeeze#batch_normalization_3/batchnorm/mul*
T0*
_output_shapes	
:�
�
.batch_normalization_3/batchnorm/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
dtype0*
_output_shapes	
:�
�
#batch_normalization_3/batchnorm/subSub.batch_normalization_3/batchnorm/ReadVariableOp%batch_normalization_3/batchnorm/mul_2*
T0*
_output_shapes	
:�
�
%batch_normalization_3/batchnorm/add_1AddV2%batch_normalization_3/batchnorm/mul_1#batch_normalization_3/batchnorm/sub*
T0*(
_output_shapes
:����������
g
batch_normalization_3/ShapeShapedense_3/Relu*
T0*
out_type0*
_output_shapes
:
s
)batch_normalization_3/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
u
+batch_normalization_3/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
u
+batch_normalization_3/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
#batch_normalization_3/strided_sliceStridedSlicebatch_normalization_3/Shape)batch_normalization_3/strided_slice/stack+batch_normalization_3/strided_slice/stack_1+batch_normalization_3/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
�
!batch_normalization_3/Rank/packedPack#batch_normalization_3/strided_slice*
T0*

axis *
N*
_output_shapes
:
\
batch_normalization_3/RankConst*
value	B :*
dtype0*
_output_shapes
: 
c
!batch_normalization_3/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
c
!batch_normalization_3/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
batch_normalization_3/rangeRange!batch_normalization_3/range/startbatch_normalization_3/Rank!batch_normalization_3/range/delta*

Tidx0*
_output_shapes
:
�
 batch_normalization_3/Prod/inputPack#batch_normalization_3/strided_slice*
T0*

axis *
N*
_output_shapes
:
�
batch_normalization_3/ProdProd batch_normalization_3/Prod/inputbatch_normalization_3/range*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
~
batch_normalization_3/CastCastbatch_normalization_3/Prod*

SrcT0*
Truncate( *

DstT0*
_output_shapes
: 
`
batch_normalization_3/sub/yConst*
valueB
 *� �?*
dtype0*
_output_shapes
: 
z
batch_normalization_3/subSubbatch_normalization_3/Castbatch_normalization_3/sub/y*
T0*
_output_shapes
: 
�
batch_normalization_3/truedivRealDivbatch_normalization_3/Castbatch_normalization_3/sub*
T0*
_output_shapes
: 
�
batch_normalization_3/mulMul'batch_normalization_3/moments/Squeeze_1batch_normalization_3/truediv*
T0*
_output_shapes	
:�
�
batch_normalization_3/Const_4Const*4
_class*
(&loc:@batch_normalization_3/moving_mean*
valueB
 *
�#<*
dtype0*
_output_shapes
: 
�
$batch_normalization_3/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
dtype0*
_output_shapes	
:�
�
batch_normalization_3/sub_1Sub$batch_normalization_3/ReadVariableOp%batch_normalization_3/moments/Squeeze*
_output_shapes	
:�*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean
�
batch_normalization_3/mul_1Mulbatch_normalization_3/sub_1batch_normalization_3/Const_4*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
_output_shapes	
:�
�
)batch_normalization_3/AssignSubVariableOpAssignSubVariableOp!batch_normalization_3/moving_meanbatch_normalization_3/mul_1*4
_class*
(&loc:@batch_normalization_3/moving_mean*
dtype0
�
&batch_normalization_3/ReadVariableOp_1ReadVariableOp!batch_normalization_3/moving_mean*^batch_normalization_3/AssignSubVariableOp*4
_class*
(&loc:@batch_normalization_3/moving_mean*
dtype0*
_output_shapes	
:�
�
batch_normalization_3/Const_5Const*8
_class.
,*loc:@batch_normalization_3/moving_variance*
valueB
 *
�#<*
dtype0*
_output_shapes
: 
�
&batch_normalization_3/ReadVariableOp_2ReadVariableOp%batch_normalization_3/moving_variance*
dtype0*
_output_shapes	
:�
�
batch_normalization_3/sub_2Sub&batch_normalization_3/ReadVariableOp_2batch_normalization_3/mul*
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes	
:�
�
batch_normalization_3/mul_2Mulbatch_normalization_3/sub_2batch_normalization_3/Const_5*
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes	
:�
�
+batch_normalization_3/AssignSubVariableOp_1AssignSubVariableOp%batch_normalization_3/moving_variancebatch_normalization_3/mul_2*8
_class.
,*loc:@batch_normalization_3/moving_variance*
dtype0
�
&batch_normalization_3/ReadVariableOp_3ReadVariableOp%batch_normalization_3/moving_variance,^batch_normalization_3/AssignSubVariableOp_1*
dtype0*
_output_shapes	
:�*8
_class.
,*loc:@batch_normalization_3/moving_variance
z
!batch_normalization_3/cond/SwitchSwitchkeras_learning_phasekeras_learning_phase*
_output_shapes
: : *
T0

u
#batch_normalization_3/cond/switch_tIdentity#batch_normalization_3/cond/Switch:1*
T0
*
_output_shapes
: 
s
#batch_normalization_3/cond/switch_fIdentity!batch_normalization_3/cond/Switch*
T0
*
_output_shapes
: 
e
"batch_normalization_3/cond/pred_idIdentitykeras_learning_phase*
_output_shapes
: *
T0

�
#batch_normalization_3/cond/Switch_1Switch%batch_normalization_3/batchnorm/add_1"batch_normalization_3/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1*<
_output_shapes*
(:����������:����������
�
3batch_normalization_3/cond/batchnorm/ReadVariableOpReadVariableOp:batch_normalization_3/cond/batchnorm/ReadVariableOp/Switch*
dtype0*
_output_shapes	
:�
�
:batch_normalization_3/cond/batchnorm/ReadVariableOp/SwitchSwitch%batch_normalization_3/moving_variance"batch_normalization_3/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes
: : 
�
*batch_normalization_3/cond/batchnorm/add/yConst$^batch_normalization_3/cond/switch_f*
dtype0*
_output_shapes
: *
valueB
 *o�:
�
(batch_normalization_3/cond/batchnorm/addAddV23batch_normalization_3/cond/batchnorm/ReadVariableOp*batch_normalization_3/cond/batchnorm/add/y*
_output_shapes	
:�*
T0
�
*batch_normalization_3/cond/batchnorm/RsqrtRsqrt(batch_normalization_3/cond/batchnorm/add*
T0*
_output_shapes	
:�
�
7batch_normalization_3/cond/batchnorm/mul/ReadVariableOpReadVariableOp>batch_normalization_3/cond/batchnorm/mul/ReadVariableOp/Switch*
dtype0*
_output_shapes	
:�
�
>batch_normalization_3/cond/batchnorm/mul/ReadVariableOp/SwitchSwitchbatch_normalization_3/gamma"batch_normalization_3/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_3/gamma*
_output_shapes
: : 
�
(batch_normalization_3/cond/batchnorm/mulMul*batch_normalization_3/cond/batchnorm/Rsqrt7batch_normalization_3/cond/batchnorm/mul/ReadVariableOp*
_output_shapes	
:�*
T0
�
*batch_normalization_3/cond/batchnorm/mul_1Mul1batch_normalization_3/cond/batchnorm/mul_1/Switch(batch_normalization_3/cond/batchnorm/mul*(
_output_shapes
:����������*
T0
�
1batch_normalization_3/cond/batchnorm/mul_1/SwitchSwitchdense_3/Relu"batch_normalization_3/cond/pred_id*
T0*
_class
loc:@dense_3/Relu*<
_output_shapes*
(:����������:����������
�
5batch_normalization_3/cond/batchnorm/ReadVariableOp_1ReadVariableOp<batch_normalization_3/cond/batchnorm/ReadVariableOp_1/Switch*
dtype0*
_output_shapes	
:�
�
<batch_normalization_3/cond/batchnorm/ReadVariableOp_1/SwitchSwitch!batch_normalization_3/moving_mean"batch_normalization_3/cond/pred_id*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
_output_shapes
: : 
�
*batch_normalization_3/cond/batchnorm/mul_2Mul5batch_normalization_3/cond/batchnorm/ReadVariableOp_1(batch_normalization_3/cond/batchnorm/mul*
T0*
_output_shapes	
:�
�
5batch_normalization_3/cond/batchnorm/ReadVariableOp_2ReadVariableOp<batch_normalization_3/cond/batchnorm/ReadVariableOp_2/Switch*
dtype0*
_output_shapes	
:�
�
<batch_normalization_3/cond/batchnorm/ReadVariableOp_2/SwitchSwitchbatch_normalization_3/beta"batch_normalization_3/cond/pred_id*
_output_shapes
: : *
T0*-
_class#
!loc:@batch_normalization_3/beta
�
(batch_normalization_3/cond/batchnorm/subSub5batch_normalization_3/cond/batchnorm/ReadVariableOp_2*batch_normalization_3/cond/batchnorm/mul_2*
T0*
_output_shapes	
:�
�
*batch_normalization_3/cond/batchnorm/add_1AddV2*batch_normalization_3/cond/batchnorm/mul_1(batch_normalization_3/cond/batchnorm/sub*
T0*(
_output_shapes
:����������
�
 batch_normalization_3/cond/MergeMerge*batch_normalization_3/cond/batchnorm/add_1%batch_normalization_3/cond/Switch_1:1*
T0*
N**
_output_shapes
:����������: 
n
dropout_2/cond/SwitchSwitchkeras_learning_phasekeras_learning_phase*
T0
*
_output_shapes
: : 
]
dropout_2/cond/switch_tIdentitydropout_2/cond/Switch:1*
T0
*
_output_shapes
: 
[
dropout_2/cond/switch_fIdentitydropout_2/cond/Switch*
_output_shapes
: *
T0

Y
dropout_2/cond/pred_idIdentitykeras_learning_phase*
T0
*
_output_shapes
: 
z
dropout_2/cond/dropout/rateConst^dropout_2/cond/switch_t*
valueB
 *   ?*
dtype0*
_output_shapes
: 
�
dropout_2/cond/dropout/ShapeShape%dropout_2/cond/dropout/Shape/Switch:1*
T0*
out_type0*
_output_shapes
:
�
#dropout_2/cond/dropout/Shape/SwitchSwitch batch_normalization_3/cond/Mergedropout_2/cond/pred_id*
T0*3
_class)
'%loc:@batch_normalization_3/cond/Merge*<
_output_shapes*
(:����������:����������
�
)dropout_2/cond/dropout/random_uniform/minConst^dropout_2/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
�
)dropout_2/cond/dropout/random_uniform/maxConst^dropout_2/cond/switch_t*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
3dropout_2/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_2/cond/dropout/Shape*
dtype0*
seed2�ڡ*(
_output_shapes
:����������*
seed���)*
T0
�
)dropout_2/cond/dropout/random_uniform/subSub)dropout_2/cond/dropout/random_uniform/max)dropout_2/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
�
)dropout_2/cond/dropout/random_uniform/mulMul3dropout_2/cond/dropout/random_uniform/RandomUniform)dropout_2/cond/dropout/random_uniform/sub*
T0*(
_output_shapes
:����������
�
%dropout_2/cond/dropout/random_uniformAdd)dropout_2/cond/dropout/random_uniform/mul)dropout_2/cond/dropout/random_uniform/min*(
_output_shapes
:����������*
T0
{
dropout_2/cond/dropout/sub/xConst^dropout_2/cond/switch_t*
valueB
 *  �?*
dtype0*
_output_shapes
: 
}
dropout_2/cond/dropout/subSubdropout_2/cond/dropout/sub/xdropout_2/cond/dropout/rate*
T0*
_output_shapes
: 

 dropout_2/cond/dropout/truediv/xConst^dropout_2/cond/switch_t*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
dropout_2/cond/dropout/truedivRealDiv dropout_2/cond/dropout/truediv/xdropout_2/cond/dropout/sub*
T0*
_output_shapes
: 
�
#dropout_2/cond/dropout/GreaterEqualGreaterEqual%dropout_2/cond/dropout/random_uniformdropout_2/cond/dropout/rate*(
_output_shapes
:����������*
T0
�
dropout_2/cond/dropout/mulMul%dropout_2/cond/dropout/Shape/Switch:1dropout_2/cond/dropout/truediv*(
_output_shapes
:����������*
T0
�
dropout_2/cond/dropout/CastCast#dropout_2/cond/dropout/GreaterEqual*

SrcT0
*
Truncate( *

DstT0*(
_output_shapes
:����������
�
dropout_2/cond/dropout/mul_1Muldropout_2/cond/dropout/muldropout_2/cond/dropout/Cast*
T0*(
_output_shapes
:����������
�
dropout_2/cond/Switch_1Switch batch_normalization_3/cond/Mergedropout_2/cond/pred_id*
T0*3
_class)
'%loc:@batch_normalization_3/cond/Merge*<
_output_shapes*
(:����������:����������
�
dropout_2/cond/MergeMergedropout_2/cond/Switch_1dropout_2/cond/dropout/mul_1*
T0*
N**
_output_shapes
:����������: 
m
dense_4/random_uniform/shapeConst*
valueB"�   �   *
dtype0*
_output_shapes
:
_
dense_4/random_uniform/minConst*
valueB
 *q��*
dtype0*
_output_shapes
: 
_
dense_4/random_uniform/maxConst*
valueB
 *q�>*
dtype0*
_output_shapes
: 
�
$dense_4/random_uniform/RandomUniformRandomUniformdense_4/random_uniform/shape*
dtype0*
seed2�* 
_output_shapes
:
��*
seed���)*
T0
z
dense_4/random_uniform/subSubdense_4/random_uniform/maxdense_4/random_uniform/min*
T0*
_output_shapes
: 
�
dense_4/random_uniform/mulMul$dense_4/random_uniform/RandomUniformdense_4/random_uniform/sub*
T0* 
_output_shapes
:
��
�
dense_4/random_uniformAdddense_4/random_uniform/muldense_4/random_uniform/min*
T0* 
_output_shapes
:
��
�
dense_4/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shared_namedense_4/kernel*!
_class
loc:@dense_4/kernel*
	container *
shape:
��
m
/dense_4/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_4/kernel*
_output_shapes
: 
^
dense_4/kernel/AssignAssignVariableOpdense_4/kerneldense_4/random_uniform*
dtype0
s
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
dtype0* 
_output_shapes
:
��
\
dense_4/ConstConst*
valueB�*    *
dtype0*
_output_shapes	
:�
�
dense_4/biasVarHandleOp*
shared_namedense_4/bias*
_class
loc:@dense_4/bias*
	container *
shape:�*
dtype0*
_output_shapes
: 
i
-dense_4/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_4/bias*
_output_shapes
: 
Q
dense_4/bias/AssignAssignVariableOpdense_4/biasdense_4/Const*
dtype0
j
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
dtype0*
_output_shapes	
:�
n
dense_4/MatMul/ReadVariableOpReadVariableOpdense_4/kernel*
dtype0* 
_output_shapes
:
��
�
dense_4/MatMulMatMuldropout_2/cond/Mergedense_4/MatMul/ReadVariableOp*
transpose_b( *
T0*
transpose_a( *(
_output_shapes
:����������
h
dense_4/BiasAdd/ReadVariableOpReadVariableOpdense_4/bias*
dtype0*
_output_shapes	
:�
�
dense_4/BiasAddBiasAdddense_4/MatMuldense_4/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*(
_output_shapes
:����������
X
dense_4/ReluReludense_4/BiasAdd*
T0*(
_output_shapes
:����������
j
batch_normalization_4/ConstConst*
valueB�*  �?*
dtype0*
_output_shapes	
:�
�
batch_normalization_4/gammaVarHandleOp*
dtype0*
_output_shapes
: *,
shared_namebatch_normalization_4/gamma*.
_class$
" loc:@batch_normalization_4/gamma*
	container *
shape:�
�
<batch_normalization_4/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_4/gamma*
_output_shapes
: 
}
"batch_normalization_4/gamma/AssignAssignVariableOpbatch_normalization_4/gammabatch_normalization_4/Const*
dtype0
�
/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_4/gamma*
dtype0*
_output_shapes	
:�
l
batch_normalization_4/Const_1Const*
dtype0*
_output_shapes	
:�*
valueB�*    
�
batch_normalization_4/betaVarHandleOp*+
shared_namebatch_normalization_4/beta*-
_class#
!loc:@batch_normalization_4/beta*
	container *
shape:�*
dtype0*
_output_shapes
: 
�
;batch_normalization_4/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_4/beta*
_output_shapes
: 
}
!batch_normalization_4/beta/AssignAssignVariableOpbatch_normalization_4/betabatch_normalization_4/Const_1*
dtype0
�
.batch_normalization_4/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_4/beta*
dtype0*
_output_shapes	
:�
l
batch_normalization_4/Const_2Const*
dtype0*
_output_shapes	
:�*
valueB�*    
�
!batch_normalization_4/moving_meanVarHandleOp*2
shared_name#!batch_normalization_4/moving_mean*4
_class*
(&loc:@batch_normalization_4/moving_mean*
	container *
shape:�*
dtype0*
_output_shapes
: 
�
Bbatch_normalization_4/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp!batch_normalization_4/moving_mean*
_output_shapes
: 
�
(batch_normalization_4/moving_mean/AssignAssignVariableOp!batch_normalization_4/moving_meanbatch_normalization_4/Const_2*
dtype0
�
5batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
dtype0*
_output_shapes	
:�
l
batch_normalization_4/Const_3Const*
valueB�*  �?*
dtype0*
_output_shapes	
:�
�
%batch_normalization_4/moving_varianceVarHandleOp*
dtype0*
_output_shapes
: *6
shared_name'%batch_normalization_4/moving_variance*8
_class.
,*loc:@batch_normalization_4/moving_variance*
	container *
shape:�
�
Fbatch_normalization_4/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp%batch_normalization_4/moving_variance*
_output_shapes
: 
�
,batch_normalization_4/moving_variance/AssignAssignVariableOp%batch_normalization_4/moving_variancebatch_normalization_4/Const_3*
dtype0
�
9batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*
dtype0*
_output_shapes	
:�
~
4batch_normalization_4/moments/mean/reduction_indicesConst*
valueB: *
dtype0*
_output_shapes
:
�
"batch_normalization_4/moments/meanMeandense_4/Relu4batch_normalization_4/moments/mean/reduction_indices*
T0*
_output_shapes
:	�*

Tidx0*
	keep_dims(
�
*batch_normalization_4/moments/StopGradientStopGradient"batch_normalization_4/moments/mean*
T0*
_output_shapes
:	�
�
/batch_normalization_4/moments/SquaredDifferenceSquaredDifferencedense_4/Relu*batch_normalization_4/moments/StopGradient*
T0*(
_output_shapes
:����������
�
8batch_normalization_4/moments/variance/reduction_indicesConst*
valueB: *
dtype0*
_output_shapes
:
�
&batch_normalization_4/moments/varianceMean/batch_normalization_4/moments/SquaredDifference8batch_normalization_4/moments/variance/reduction_indices*
_output_shapes
:	�*

Tidx0*
	keep_dims(*
T0
�
%batch_normalization_4/moments/SqueezeSqueeze"batch_normalization_4/moments/mean*
squeeze_dims
 *
T0*
_output_shapes	
:�
�
'batch_normalization_4/moments/Squeeze_1Squeeze&batch_normalization_4/moments/variance*
_output_shapes	
:�*
squeeze_dims
 *
T0
j
%batch_normalization_4/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
#batch_normalization_4/batchnorm/addAddV2'batch_normalization_4/moments/Squeeze_1%batch_normalization_4/batchnorm/add/y*
T0*
_output_shapes	
:�
y
%batch_normalization_4/batchnorm/RsqrtRsqrt#batch_normalization_4/batchnorm/add*
T0*
_output_shapes	
:�
�
2batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOpbatch_normalization_4/gamma*
dtype0*
_output_shapes	
:�
�
#batch_normalization_4/batchnorm/mulMul%batch_normalization_4/batchnorm/Rsqrt2batch_normalization_4/batchnorm/mul/ReadVariableOp*
T0*
_output_shapes	
:�
�
%batch_normalization_4/batchnorm/mul_1Muldense_4/Relu#batch_normalization_4/batchnorm/mul*
T0*(
_output_shapes
:����������
�
%batch_normalization_4/batchnorm/mul_2Mul%batch_normalization_4/moments/Squeeze#batch_normalization_4/batchnorm/mul*
T0*
_output_shapes	
:�
�
.batch_normalization_4/batchnorm/ReadVariableOpReadVariableOpbatch_normalization_4/beta*
dtype0*
_output_shapes	
:�
�
#batch_normalization_4/batchnorm/subSub.batch_normalization_4/batchnorm/ReadVariableOp%batch_normalization_4/batchnorm/mul_2*
T0*
_output_shapes	
:�
�
%batch_normalization_4/batchnorm/add_1AddV2%batch_normalization_4/batchnorm/mul_1#batch_normalization_4/batchnorm/sub*
T0*(
_output_shapes
:����������
g
batch_normalization_4/ShapeShapedense_4/Relu*
_output_shapes
:*
T0*
out_type0
s
)batch_normalization_4/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
u
+batch_normalization_4/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
u
+batch_normalization_4/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
#batch_normalization_4/strided_sliceStridedSlicebatch_normalization_4/Shape)batch_normalization_4/strided_slice/stack+batch_normalization_4/strided_slice/stack_1+batch_normalization_4/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
�
!batch_normalization_4/Rank/packedPack#batch_normalization_4/strided_slice*
T0*

axis *
N*
_output_shapes
:
\
batch_normalization_4/RankConst*
value	B :*
dtype0*
_output_shapes
: 
c
!batch_normalization_4/range/startConst*
dtype0*
_output_shapes
: *
value	B : 
c
!batch_normalization_4/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
batch_normalization_4/rangeRange!batch_normalization_4/range/startbatch_normalization_4/Rank!batch_normalization_4/range/delta*

Tidx0*
_output_shapes
:
�
 batch_normalization_4/Prod/inputPack#batch_normalization_4/strided_slice*
T0*

axis *
N*
_output_shapes
:
�
batch_normalization_4/ProdProd batch_normalization_4/Prod/inputbatch_normalization_4/range*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
~
batch_normalization_4/CastCastbatch_normalization_4/Prod*

SrcT0*
Truncate( *

DstT0*
_output_shapes
: 
`
batch_normalization_4/sub/yConst*
valueB
 *� �?*
dtype0*
_output_shapes
: 
z
batch_normalization_4/subSubbatch_normalization_4/Castbatch_normalization_4/sub/y*
_output_shapes
: *
T0
�
batch_normalization_4/truedivRealDivbatch_normalization_4/Castbatch_normalization_4/sub*
_output_shapes
: *
T0
�
batch_normalization_4/mulMul'batch_normalization_4/moments/Squeeze_1batch_normalization_4/truediv*
_output_shapes	
:�*
T0
�
batch_normalization_4/Const_4Const*4
_class*
(&loc:@batch_normalization_4/moving_mean*
valueB
 *
�#<*
dtype0*
_output_shapes
: 
�
$batch_normalization_4/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
dtype0*
_output_shapes	
:�
�
batch_normalization_4/sub_1Sub$batch_normalization_4/ReadVariableOp%batch_normalization_4/moments/Squeeze*
_output_shapes	
:�*
T0*4
_class*
(&loc:@batch_normalization_4/moving_mean
�
batch_normalization_4/mul_1Mulbatch_normalization_4/sub_1batch_normalization_4/Const_4*
_output_shapes	
:�*
T0*4
_class*
(&loc:@batch_normalization_4/moving_mean
�
)batch_normalization_4/AssignSubVariableOpAssignSubVariableOp!batch_normalization_4/moving_meanbatch_normalization_4/mul_1*4
_class*
(&loc:@batch_normalization_4/moving_mean*
dtype0
�
&batch_normalization_4/ReadVariableOp_1ReadVariableOp!batch_normalization_4/moving_mean*^batch_normalization_4/AssignSubVariableOp*4
_class*
(&loc:@batch_normalization_4/moving_mean*
dtype0*
_output_shapes	
:�
�
batch_normalization_4/Const_5Const*8
_class.
,*loc:@batch_normalization_4/moving_variance*
valueB
 *
�#<*
dtype0*
_output_shapes
: 
�
&batch_normalization_4/ReadVariableOp_2ReadVariableOp%batch_normalization_4/moving_variance*
dtype0*
_output_shapes	
:�
�
batch_normalization_4/sub_2Sub&batch_normalization_4/ReadVariableOp_2batch_normalization_4/mul*
_output_shapes	
:�*
T0*8
_class.
,*loc:@batch_normalization_4/moving_variance
�
batch_normalization_4/mul_2Mulbatch_normalization_4/sub_2batch_normalization_4/Const_5*
T0*8
_class.
,*loc:@batch_normalization_4/moving_variance*
_output_shapes	
:�
�
+batch_normalization_4/AssignSubVariableOp_1AssignSubVariableOp%batch_normalization_4/moving_variancebatch_normalization_4/mul_2*8
_class.
,*loc:@batch_normalization_4/moving_variance*
dtype0
�
&batch_normalization_4/ReadVariableOp_3ReadVariableOp%batch_normalization_4/moving_variance,^batch_normalization_4/AssignSubVariableOp_1*
dtype0*
_output_shapes	
:�*8
_class.
,*loc:@batch_normalization_4/moving_variance
z
!batch_normalization_4/cond/SwitchSwitchkeras_learning_phasekeras_learning_phase*
T0
*
_output_shapes
: : 
u
#batch_normalization_4/cond/switch_tIdentity#batch_normalization_4/cond/Switch:1*
T0
*
_output_shapes
: 
s
#batch_normalization_4/cond/switch_fIdentity!batch_normalization_4/cond/Switch*
T0
*
_output_shapes
: 
e
"batch_normalization_4/cond/pred_idIdentitykeras_learning_phase*
T0
*
_output_shapes
: 
�
#batch_normalization_4/cond/Switch_1Switch%batch_normalization_4/batchnorm/add_1"batch_normalization_4/cond/pred_id*<
_output_shapes*
(:����������:����������*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1
�
3batch_normalization_4/cond/batchnorm/ReadVariableOpReadVariableOp:batch_normalization_4/cond/batchnorm/ReadVariableOp/Switch*
dtype0*
_output_shapes	
:�
�
:batch_normalization_4/cond/batchnorm/ReadVariableOp/SwitchSwitch%batch_normalization_4/moving_variance"batch_normalization_4/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_4/moving_variance*
_output_shapes
: : 
�
*batch_normalization_4/cond/batchnorm/add/yConst$^batch_normalization_4/cond/switch_f*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
(batch_normalization_4/cond/batchnorm/addAddV23batch_normalization_4/cond/batchnorm/ReadVariableOp*batch_normalization_4/cond/batchnorm/add/y*
T0*
_output_shapes	
:�
�
*batch_normalization_4/cond/batchnorm/RsqrtRsqrt(batch_normalization_4/cond/batchnorm/add*
T0*
_output_shapes	
:�
�
7batch_normalization_4/cond/batchnorm/mul/ReadVariableOpReadVariableOp>batch_normalization_4/cond/batchnorm/mul/ReadVariableOp/Switch*
dtype0*
_output_shapes	
:�
�
>batch_normalization_4/cond/batchnorm/mul/ReadVariableOp/SwitchSwitchbatch_normalization_4/gamma"batch_normalization_4/cond/pred_id*
_output_shapes
: : *
T0*.
_class$
" loc:@batch_normalization_4/gamma
�
(batch_normalization_4/cond/batchnorm/mulMul*batch_normalization_4/cond/batchnorm/Rsqrt7batch_normalization_4/cond/batchnorm/mul/ReadVariableOp*
T0*
_output_shapes	
:�
�
*batch_normalization_4/cond/batchnorm/mul_1Mul1batch_normalization_4/cond/batchnorm/mul_1/Switch(batch_normalization_4/cond/batchnorm/mul*
T0*(
_output_shapes
:����������
�
1batch_normalization_4/cond/batchnorm/mul_1/SwitchSwitchdense_4/Relu"batch_normalization_4/cond/pred_id*
T0*
_class
loc:@dense_4/Relu*<
_output_shapes*
(:����������:����������
�
5batch_normalization_4/cond/batchnorm/ReadVariableOp_1ReadVariableOp<batch_normalization_4/cond/batchnorm/ReadVariableOp_1/Switch*
dtype0*
_output_shapes	
:�
�
<batch_normalization_4/cond/batchnorm/ReadVariableOp_1/SwitchSwitch!batch_normalization_4/moving_mean"batch_normalization_4/cond/pred_id*
_output_shapes
: : *
T0*4
_class*
(&loc:@batch_normalization_4/moving_mean
�
*batch_normalization_4/cond/batchnorm/mul_2Mul5batch_normalization_4/cond/batchnorm/ReadVariableOp_1(batch_normalization_4/cond/batchnorm/mul*
T0*
_output_shapes	
:�
�
5batch_normalization_4/cond/batchnorm/ReadVariableOp_2ReadVariableOp<batch_normalization_4/cond/batchnorm/ReadVariableOp_2/Switch*
dtype0*
_output_shapes	
:�
�
<batch_normalization_4/cond/batchnorm/ReadVariableOp_2/SwitchSwitchbatch_normalization_4/beta"batch_normalization_4/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_4/beta*
_output_shapes
: : 
�
(batch_normalization_4/cond/batchnorm/subSub5batch_normalization_4/cond/batchnorm/ReadVariableOp_2*batch_normalization_4/cond/batchnorm/mul_2*
T0*
_output_shapes	
:�
�
*batch_normalization_4/cond/batchnorm/add_1AddV2*batch_normalization_4/cond/batchnorm/mul_1(batch_normalization_4/cond/batchnorm/sub*
T0*(
_output_shapes
:����������
�
 batch_normalization_4/cond/MergeMerge*batch_normalization_4/cond/batchnorm/add_1%batch_normalization_4/cond/Switch_1:1*
T0*
N**
_output_shapes
:����������: 
m
dense_5/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"�   +   
_
dense_5/random_uniform/minConst*
valueB
 *�?�*
dtype0*
_output_shapes
: 
_
dense_5/random_uniform/maxConst*
valueB
 *�?>*
dtype0*
_output_shapes
: 
�
$dense_5/random_uniform/RandomUniformRandomUniformdense_5/random_uniform/shape*
seed���)*
T0*
dtype0*
seed2�ب*
_output_shapes
:	�+
z
dense_5/random_uniform/subSubdense_5/random_uniform/maxdense_5/random_uniform/min*
T0*
_output_shapes
: 
�
dense_5/random_uniform/mulMul$dense_5/random_uniform/RandomUniformdense_5/random_uniform/sub*
T0*
_output_shapes
:	�+

dense_5/random_uniformAdddense_5/random_uniform/muldense_5/random_uniform/min*
_output_shapes
:	�+*
T0
�
dense_5/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shared_namedense_5/kernel*!
_class
loc:@dense_5/kernel*
	container *
shape:	�+
m
/dense_5/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_5/kernel*
_output_shapes
: 
^
dense_5/kernel/AssignAssignVariableOpdense_5/kerneldense_5/random_uniform*
dtype0
r
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
dtype0*
_output_shapes
:	�+
Z
dense_5/ConstConst*
valueB+*    *
dtype0*
_output_shapes
:+
�
dense_5/biasVarHandleOp*
dtype0*
_output_shapes
: *
shared_namedense_5/bias*
_class
loc:@dense_5/bias*
	container *
shape:+
i
-dense_5/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_5/bias*
_output_shapes
: 
Q
dense_5/bias/AssignAssignVariableOpdense_5/biasdense_5/Const*
dtype0
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
dtype0*
_output_shapes
:+
m
dense_5/MatMul/ReadVariableOpReadVariableOpdense_5/kernel*
dtype0*
_output_shapes
:	�+
�
dense_5/MatMulMatMul batch_normalization_4/cond/Mergedense_5/MatMul/ReadVariableOp*
T0*
transpose_a( *'
_output_shapes
:���������+*
transpose_b( 
g
dense_5/BiasAdd/ReadVariableOpReadVariableOpdense_5/bias*
dtype0*
_output_shapes
:+
�
dense_5/BiasAddBiasAdddense_5/MatMuldense_5/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*'
_output_shapes
:���������+
]
dense_5/SoftmaxSoftmaxdense_5/BiasAdd*
T0*'
_output_shapes
:���������+
�
)Adam/iterations/Initializer/initial_valueConst*"
_class
loc:@Adam/iterations*
value	B	 R *
dtype0	*
_output_shapes
: 
�
Adam/iterationsVarHandleOp*
dtype0	*
_output_shapes
: * 
shared_nameAdam/iterations*"
_class
loc:@Adam/iterations*
	container *
shape: 
o
0Adam/iterations/IsInitialized/VarIsInitializedOpVarIsInitializedOpAdam/iterations*
_output_shapes
: 
s
Adam/iterations/AssignAssignVariableOpAdam/iterations)Adam/iterations/Initializer/initial_value*
dtype0	
k
#Adam/iterations/Read/ReadVariableOpReadVariableOpAdam/iterations*
dtype0	*
_output_shapes
: 
�
,Adam/learning_rate/Initializer/initial_valueConst*
dtype0*
_output_shapes
: *%
_class
loc:@Adam/learning_rate*
valueB
 *o�:
�
Adam/learning_rateVarHandleOp*#
shared_nameAdam/learning_rate*%
_class
loc:@Adam/learning_rate*
	container *
shape: *
dtype0*
_output_shapes
: 
u
3Adam/learning_rate/IsInitialized/VarIsInitializedOpVarIsInitializedOpAdam/learning_rate*
_output_shapes
: 
|
Adam/learning_rate/AssignAssignVariableOpAdam/learning_rate,Adam/learning_rate/Initializer/initial_value*
dtype0
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
dtype0*
_output_shapes
: 
�
%Adam/beta_1/Initializer/initial_valueConst*
_class
loc:@Adam/beta_1*
valueB
 *fff?*
dtype0*
_output_shapes
: 
�
Adam/beta_1VarHandleOp*
dtype0*
_output_shapes
: *
shared_nameAdam/beta_1*
_class
loc:@Adam/beta_1*
	container *
shape: 
g
,Adam/beta_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpAdam/beta_1*
_output_shapes
: 
g
Adam/beta_1/AssignAssignVariableOpAdam/beta_1%Adam/beta_1/Initializer/initial_value*
dtype0
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
�
%Adam/beta_2/Initializer/initial_valueConst*
dtype0*
_output_shapes
: *
_class
loc:@Adam/beta_2*
valueB
 *w�?
�
Adam/beta_2VarHandleOp*
dtype0*
_output_shapes
: *
shared_nameAdam/beta_2*
_class
loc:@Adam/beta_2*
	container *
shape: 
g
,Adam/beta_2/IsInitialized/VarIsInitializedOpVarIsInitializedOpAdam/beta_2*
_output_shapes
: 
g
Adam/beta_2/AssignAssignVariableOpAdam/beta_2%Adam/beta_2/Initializer/initial_value*
dtype0
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
�
$Adam/decay/Initializer/initial_valueConst*
dtype0*
_output_shapes
: *
_class
loc:@Adam/decay*
valueB
 *    
�

Adam/decayVarHandleOp*
_class
loc:@Adam/decay*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name
Adam/decay
e
+Adam/decay/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Adam/decay*
_output_shapes
: 
d
Adam/decay/AssignAssignVariableOp
Adam/decay$Adam/decay/Initializer/initial_value*
dtype0
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
dtype0*
_output_shapes
: 
�
dense_5_targetPlaceholder*
dtype0*0
_output_shapes
:������������������*%
shape:������������������
q
dense_5_sample_weightsPlaceholder*
dtype0*#
_output_shapes
:���������*
shape:���������
J
ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
totalVarHandleOp*
_class

loc:@total*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_nametotal
[
&total/IsInitialized/VarIsInitializedOpVarIsInitializedOptotal*
_output_shapes
: 
;
total/AssignAssignVariableOptotalConst*
dtype0
W
total/Read/ReadVariableOpReadVariableOptotal*
dtype0*
_output_shapes
: 
L
Const_1Const*
dtype0*
_output_shapes
: *
valueB
 *    
�
countVarHandleOp*
dtype0*
_output_shapes
: *
shared_namecount*
_class

loc:@count*
	container *
shape: 
[
&count/IsInitialized/VarIsInitializedOpVarIsInitializedOpcount*
_output_shapes
: 
=
count/AssignAssignVariableOpcountConst_1*
dtype0
W
count/Read/ReadVariableOpReadVariableOpcount*
dtype0*
_output_shapes
: 
l
!metrics/accuracy/ArgMax/dimensionConst*
dtype0*
_output_shapes
: *
valueB :
���������
�
metrics/accuracy/ArgMaxArgMaxdense_5_target!metrics/accuracy/ArgMax/dimension*
T0*
output_type0	*#
_output_shapes
:���������*

Tidx0
n
#metrics/accuracy/ArgMax_1/dimensionConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
metrics/accuracy/ArgMax_1ArgMaxdense_5/Softmax#metrics/accuracy/ArgMax_1/dimension*
T0*
output_type0	*#
_output_shapes
:���������*

Tidx0
�
metrics/accuracy/EqualEqualmetrics/accuracy/ArgMaxmetrics/accuracy/ArgMax_1*
incompatible_shape_error(*
T0	*#
_output_shapes
:���������
�
metrics/accuracy/CastCastmetrics/accuracy/Equal*

SrcT0
*
Truncate( *

DstT0*#
_output_shapes
:���������
`
metrics/accuracy/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
�
metrics/accuracy/SumSummetrics/accuracy/Castmetrics/accuracy/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
e
$metrics/accuracy/AssignAddVariableOpAssignAddVariableOptotalmetrics/accuracy/Sum*
dtype0
�
metrics/accuracy/ReadVariableOpReadVariableOptotal%^metrics/accuracy/AssignAddVariableOp*
dtype0*
_output_shapes
: 
e
metrics/accuracy/SizeSizemetrics/accuracy/Cast*
_output_shapes
: *
T0*
out_type0
v
metrics/accuracy/Cast_1Castmetrics/accuracy/Size*

SrcT0*
Truncate( *

DstT0*
_output_shapes
: 
j
&metrics/accuracy/AssignAddVariableOp_1AssignAddVariableOpcountmetrics/accuracy/Cast_1*
dtype0
�
!metrics/accuracy/ReadVariableOp_1ReadVariableOpcount'^metrics/accuracy/AssignAddVariableOp_1*
dtype0*
_output_shapes
: 
�
!metrics/accuracy/ReadVariableOp_2ReadVariableOptotal%^metrics/accuracy/AssignAddVariableOp'^metrics/accuracy/AssignAddVariableOp_1*
dtype0*
_output_shapes
: 
�
'metrics/accuracy/truediv/ReadVariableOpReadVariableOpcount%^metrics/accuracy/AssignAddVariableOp'^metrics/accuracy/AssignAddVariableOp_1*
dtype0*
_output_shapes
: 
�
metrics/accuracy/truedivRealDiv!metrics/accuracy/ReadVariableOp_2'metrics/accuracy/truediv/ReadVariableOp*
T0*
_output_shapes
: 
`
metrics/accuracy/IdentityIdentitymetrics/accuracy/truediv*
T0*
_output_shapes
: 
�
Qloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/RankConst*
value	B :*
dtype0*
_output_shapes
: 
�
Rloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/ShapeShapedense_5/BiasAdd*
_output_shapes
:*
T0*
out_type0
�
Sloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
�
Tloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Shape_1Shapedense_5/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
Rloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
Ploss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/SubSubSloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Rank_1Rloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Sub/y*
T0*
_output_shapes
: 
�
Xloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Slice/beginPackPloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Sub*
N*
_output_shapes
:*
T0*

axis 
�
Wloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Slice/sizeConst*
dtype0*
_output_shapes
:*
valueB:
�
Rloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/SliceSliceTloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Shape_1Xloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Slice/beginWloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Slice/size*
T0*
Index0*
_output_shapes
:
�
\loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/concat/values_0Const*
dtype0*
_output_shapes
:*
valueB:
���������
�
Xloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
Sloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/concatConcatV2\loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/concat/values_0Rloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/SliceXloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
�
Tloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/ReshapeReshapedense_5/BiasAddSloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/concat*
T0*
Tshape0*0
_output_shapes
:������������������
�
Sloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Rank_2Const*
dtype0*
_output_shapes
: *
value	B :
�
Tloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Shape_2Shapedense_5_target*
T0*
out_type0*
_output_shapes
:
�
Tloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Sub_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
Rloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Sub_1SubSloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Rank_2Tloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Sub_1/y*
T0*
_output_shapes
: 
�
Zloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Slice_1/beginPackRloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Sub_1*
T0*

axis *
N*
_output_shapes
:
�
Yloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Slice_1/sizeConst*
dtype0*
_output_shapes
:*
valueB:
�
Tloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Slice_1SliceTloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Shape_2Zloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Slice_1/beginYloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Slice_1/size*
_output_shapes
:*
T0*
Index0
�
^loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/concat_1/values_0Const*
dtype0*
_output_shapes
:*
valueB:
���������
�
Zloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Uloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/concat_1ConcatV2^loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/concat_1/values_0Tloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Slice_1Zloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/concat_1/axis*
N*
_output_shapes
:*

Tidx0*
T0
�
Vloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape_1Reshapedense_5_targetUloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/concat_1*0
_output_shapes
:������������������*
T0*
Tshape0
�
Lloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogitsTloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/ReshapeVloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape_1*
T0*?
_output_shapes-
+:���������:������������������
�
Tloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
Rloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Sub_2SubQloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/RankTloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Sub_2/y*
T0*
_output_shapes
: 
�
Zloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Slice_2/beginConst*
valueB: *
dtype0*
_output_shapes
:
�
Yloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Slice_2/sizePackRloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Sub_2*
T0*

axis *
N*
_output_shapes
:
�
Tloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Slice_2SliceRloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/ShapeZloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Slice_2/beginYloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Slice_2/size*
T0*
Index0*
_output_shapes
:
�
Vloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape_2ReshapeLloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logitsTloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Slice_2*
T0*
Tshape0*#
_output_shapes
:���������
�
<loss/dense_5_loss/categorical_crossentropy/weighted_loss/mulMuldense_5_sample_weightsVloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape_2*
T0*#
_output_shapes
:���������
�
>loss/dense_5_loss/categorical_crossentropy/weighted_loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
<loss/dense_5_loss/categorical_crossentropy/weighted_loss/SumSum<loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul>loss/dense_5_loss/categorical_crossentropy/weighted_loss/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
Jloss/dense_5_loss/categorical_crossentropy/weighted_loss/num_elements/SizeSize<loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul*
_output_shapes
: *
T0*
out_type0
�
Jloss/dense_5_loss/categorical_crossentropy/weighted_loss/num_elements/CastCastJloss/dense_5_loss/categorical_crossentropy/weighted_loss/num_elements/Size*

SrcT0*
Truncate( *

DstT0*
_output_shapes
: 
�
@loss/dense_5_loss/categorical_crossentropy/weighted_loss/truedivRealDiv<loss/dense_5_loss/categorical_crossentropy/weighted_loss/SumJloss/dense_5_loss/categorical_crossentropy/weighted_loss/num_elements/Cast*
_output_shapes
: *
T0
O

loss/mul/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
~
loss/mulMul
loss/mul/x@loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv*
T0*
_output_shapes
: 
J
Const_2Const*
valueB *
dtype0*
_output_shapes
: 
]
MeanMeanloss/mulConst_2*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
y
training/Adam/gradients/ShapeConst*
dtype0*
_output_shapes
: *
_class
	loc:@Mean*
valueB 

!training/Adam/gradients/grad_ys_0Const*
_class
	loc:@Mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
training/Adam/gradients/FillFilltraining/Adam/gradients/Shape!training/Adam/gradients/grad_ys_0*
T0*
_class
	loc:@Mean*

index_type0*
_output_shapes
: 
�
/training/Adam/gradients/Mean_grad/Reshape/shapeConst*
dtype0*
_output_shapes
: *
_class
	loc:@Mean*
valueB 
�
)training/Adam/gradients/Mean_grad/ReshapeReshapetraining/Adam/gradients/Fill/training/Adam/gradients/Mean_grad/Reshape/shape*
T0*
_class
	loc:@Mean*
Tshape0*
_output_shapes
: 
�
'training/Adam/gradients/Mean_grad/ConstConst*
dtype0*
_output_shapes
: *
_class
	loc:@Mean*
valueB 
�
&training/Adam/gradients/Mean_grad/TileTile)training/Adam/gradients/Mean_grad/Reshape'training/Adam/gradients/Mean_grad/Const*
T0*
_class
	loc:@Mean*
_output_shapes
: *

Tmultiples0
�
)training/Adam/gradients/Mean_grad/Const_1Const*
dtype0*
_output_shapes
: *
_class
	loc:@Mean*
valueB
 *  �?
�
)training/Adam/gradients/Mean_grad/truedivRealDiv&training/Adam/gradients/Mean_grad/Tile)training/Adam/gradients/Mean_grad/Const_1*
T0*
_class
	loc:@Mean*
_output_shapes
: 
�
)training/Adam/gradients/loss/mul_grad/MulMul)training/Adam/gradients/Mean_grad/truediv@loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv*
T0*
_class
loc:@loss/mul*
_output_shapes
: 
�
+training/Adam/gradients/loss/mul_grad/Mul_1Mul)training/Adam/gradients/Mean_grad/truediv
loss/mul/x*
T0*
_class
loc:@loss/mul*
_output_shapes
: 
�
ctraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/ShapeConst*S
_classI
GEloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv*
valueB *
dtype0*
_output_shapes
: 
�
etraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/Shape_1Const*
dtype0*
_output_shapes
: *S
_classI
GEloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv*
valueB 
�
straining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsctraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/Shapeetraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/Shape_1*
T0*S
_classI
GEloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv*2
_output_shapes 
:���������:���������
�
etraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/RealDivRealDiv+training/Adam/gradients/loss/mul_grad/Mul_1Jloss/dense_5_loss/categorical_crossentropy/weighted_loss/num_elements/Cast*
_output_shapes
: *
T0*S
_classI
GEloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv
�
atraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/SumSumetraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/RealDivstraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/BroadcastGradientArgs*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0*S
_classI
GEloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv
�
etraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/ReshapeReshapeatraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/Sumctraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/Shape*
_output_shapes
: *
T0*S
_classI
GEloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv*
Tshape0
�
atraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/NegNeg<loss/dense_5_loss/categorical_crossentropy/weighted_loss/Sum*
T0*S
_classI
GEloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv*
_output_shapes
: 
�
gtraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/RealDiv_1RealDivatraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/NegJloss/dense_5_loss/categorical_crossentropy/weighted_loss/num_elements/Cast*
_output_shapes
: *
T0*S
_classI
GEloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv
�
gtraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/RealDiv_2RealDivgtraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/RealDiv_1Jloss/dense_5_loss/categorical_crossentropy/weighted_loss/num_elements/Cast*
T0*S
_classI
GEloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv*
_output_shapes
: 
�
atraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/mulMul+training/Adam/gradients/loss/mul_grad/Mul_1gtraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/RealDiv_2*
T0*S
_classI
GEloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv*
_output_shapes
: 
�
ctraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/Sum_1Sumatraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/mulutraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/BroadcastGradientArgs:1*
T0*S
_classI
GEloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv*
_output_shapes
: *
	keep_dims( *

Tidx0
�
gtraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/Reshape_1Reshapectraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/Sum_1etraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/Shape_1*
T0*S
_classI
GEloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv*
Tshape0*
_output_shapes
: 
�
gtraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/Sum_grad/Reshape/shapeConst*O
_classE
CAloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/Sum*
valueB:*
dtype0*
_output_shapes
:
�
atraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/Sum_grad/ReshapeReshapeetraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/Reshapegtraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/Sum_grad/Reshape/shape*
T0*O
_classE
CAloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/Sum*
Tshape0*
_output_shapes
:
�
_training/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/Sum_grad/ShapeShape<loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul*
T0*O
_classE
CAloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/Sum*
out_type0*
_output_shapes
:
�
^training/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/Sum_grad/TileTileatraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/Sum_grad/Reshape_training/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/Sum_grad/Shape*

Tmultiples0*
T0*O
_classE
CAloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/Sum*#
_output_shapes
:���������
�
_training/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul_grad/ShapeShapedense_5_sample_weights*
T0*O
_classE
CAloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul*
out_type0*
_output_shapes
:
�
atraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul_grad/Shape_1ShapeVloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape_2*
T0*O
_classE
CAloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul*
out_type0*
_output_shapes
:
�
otraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs_training/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul_grad/Shapeatraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul_grad/Shape_1*
T0*O
_classE
CAloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul*2
_output_shapes 
:���������:���������
�
]training/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul_grad/MulMul^training/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/Sum_grad/TileVloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape_2*
T0*O
_classE
CAloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul*#
_output_shapes
:���������
�
]training/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul_grad/SumSum]training/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul_grad/Mulotraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul_grad/BroadcastGradientArgs*
T0*O
_classE
CAloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul*
_output_shapes
:*
	keep_dims( *

Tidx0
�
atraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul_grad/ReshapeReshape]training/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul_grad/Sum_training/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul_grad/Shape*#
_output_shapes
:���������*
T0*O
_classE
CAloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul*
Tshape0
�
_training/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul_grad/Mul_1Muldense_5_sample_weights^training/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/Sum_grad/Tile*
T0*O
_classE
CAloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul*#
_output_shapes
:���������
�
_training/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul_grad/Sum_1Sum_training/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul_grad/Mul_1qtraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul_grad/BroadcastGradientArgs:1*
T0*O
_classE
CAloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul*
_output_shapes
:*
	keep_dims( *

Tidx0
�
ctraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul_grad/Reshape_1Reshape_training/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul_grad/Sum_1atraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul_grad/Shape_1*#
_output_shapes
:���������*
T0*O
_classE
CAloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul*
Tshape0
�
ytraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape_2_grad/ShapeShapeLloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits*
_output_shapes
:*
T0*i
_class_
][loc:@loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape_2*
out_type0
�
{training/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape_2_grad/ReshapeReshapectraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul_grad/Reshape_1ytraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape_2_grad/Shape*
T0*i
_class_
][loc:@loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape_2*
Tshape0*#
_output_shapes
:���������
�
"training/Adam/gradients/zeros_like	ZerosLikeNloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits:1*
T0*_
_classU
SQloc:@loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits*0
_output_shapes
:������������������
�
xtraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits_grad/ExpandDims/dimConst*_
_classU
SQloc:@loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits*
valueB :
���������*
dtype0*
_output_shapes
: 
�
ttraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits_grad/ExpandDims
ExpandDims{training/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape_2_grad/Reshapextraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits_grad/ExpandDims/dim*'
_output_shapes
:���������*

Tdim0*
T0*_
_classU
SQloc:@loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits
�
mtraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits_grad/mulMulttraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits_grad/ExpandDimsNloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits:1*0
_output_shapes
:������������������*
T0*_
_classU
SQloc:@loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits
�
ttraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits_grad/LogSoftmax
LogSoftmaxTloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape*0
_output_shapes
:������������������*
T0*_
_classU
SQloc:@loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits
�
mtraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits_grad/NegNegttraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits_grad/LogSoftmax*
T0*_
_classU
SQloc:@loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits*0
_output_shapes
:������������������
�
ztraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits_grad/ExpandDims_1/dimConst*_
_classU
SQloc:@loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits*
valueB :
���������*
dtype0*
_output_shapes
: 
�
vtraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits_grad/ExpandDims_1
ExpandDims{training/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape_2_grad/Reshapeztraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits_grad/ExpandDims_1/dim*
T0*_
_classU
SQloc:@loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits*'
_output_shapes
:���������*

Tdim0
�
otraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits_grad/mul_1Mulvtraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits_grad/ExpandDims_1mtraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits_grad/Neg*
T0*_
_classU
SQloc:@loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits*0
_output_shapes
:������������������
�
wtraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape_grad/ShapeShapedense_5/BiasAdd*
T0*g
_class]
[Yloc:@loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape*
out_type0*
_output_shapes
:
�
ytraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape_grad/ReshapeReshapemtraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits_grad/mulwtraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape_grad/Shape*'
_output_shapes
:���������+*
T0*g
_class]
[Yloc:@loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape*
Tshape0
�
8training/Adam/gradients/dense_5/BiasAdd_grad/BiasAddGradBiasAddGradytraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape_grad/Reshape*
T0*"
_class
loc:@dense_5/BiasAdd*
data_formatNHWC*
_output_shapes
:+
�
2training/Adam/gradients/dense_5/MatMul_grad/MatMulMatMulytraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape_grad/Reshapedense_5/MatMul/ReadVariableOp*
T0*!
_class
loc:@dense_5/MatMul*
transpose_a( *(
_output_shapes
:����������*
transpose_b(
�
4training/Adam/gradients/dense_5/MatMul_grad/MatMul_1MatMul batch_normalization_4/cond/Mergeytraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape_grad/Reshape*
transpose_b( *
T0*!
_class
loc:@dense_5/MatMul*
transpose_a(*
_output_shapes
:	�+
�
Gtraining/Adam/gradients/batch_normalization_4/cond/Merge_grad/cond_gradSwitch2training/Adam/gradients/dense_5/MatMul_grad/MatMul"batch_normalization_4/cond/pred_id*<
_output_shapes*
(:����������:����������*
T0*!
_class
loc:@dense_5/MatMul
�
Mtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/ShapeShape*batch_normalization_4/cond/batchnorm/mul_1*
T0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/add_1*
out_type0*
_output_shapes
:
�
Otraining/Adam/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/Shape_1Shape(batch_normalization_4/cond/batchnorm/sub*
_output_shapes
:*
T0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/add_1*
out_type0
�
]training/Adam/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsMtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/ShapeOtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/Shape_1*
T0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/add_1*2
_output_shapes 
:���������:���������
�
Ktraining/Adam/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/SumSumGtraining/Adam/gradients/batch_normalization_4/cond/Merge_grad/cond_grad]training/Adam/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/add_1
�
Otraining/Adam/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/ReshapeReshapeKtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/SumMtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/Shape*
T0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/add_1*
Tshape0*(
_output_shapes
:����������
�
Mtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/Sum_1SumGtraining/Adam/gradients/batch_normalization_4/cond/Merge_grad/cond_grad_training/Adam/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/add_1
�
Qtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/Reshape_1ReshapeMtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/Sum_1Otraining/Adam/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/Shape_1*
T0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/add_1*
Tshape0*
_output_shapes	
:�
�
training/Adam/gradients/SwitchSwitch%batch_normalization_4/batchnorm/add_1"batch_normalization_4/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1*<
_output_shapes*
(:����������:����������
�
 training/Adam/gradients/IdentityIdentitytraining/Adam/gradients/Switch*(
_output_shapes
:����������*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1
�
training/Adam/gradients/Shape_1Shapetraining/Adam/gradients/Switch*
_output_shapes
:*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1*
out_type0
�
#training/Adam/gradients/zeros/ConstConst!^training/Adam/gradients/Identity*8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/gradients/zerosFilltraining/Adam/gradients/Shape_1#training/Adam/gradients/zeros/Const*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1*

index_type0*(
_output_shapes
:����������
�
Jtraining/Adam/gradients/batch_normalization_4/cond/Switch_1_grad/cond_gradMergetraining/Adam/gradients/zerosItraining/Adam/gradients/batch_normalization_4/cond/Merge_grad/cond_grad:1*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1*
N**
_output_shapes
:����������: 
�
Mtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/ShapeShape1batch_normalization_4/cond/batchnorm/mul_1/Switch*
T0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/mul_1*
out_type0*
_output_shapes
:
�
Otraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/Shape_1Shape(batch_normalization_4/cond/batchnorm/mul*
_output_shapes
:*
T0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/mul_1*
out_type0
�
]training/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsMtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/ShapeOtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/mul_1
�
Ktraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/MulMulOtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/Reshape(batch_normalization_4/cond/batchnorm/mul*(
_output_shapes
:����������*
T0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/mul_1
�
Ktraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/SumSumKtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/Mul]training/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/mul_1
�
Otraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/ReshapeReshapeKtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/SumMtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/Shape*
T0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/mul_1*
Tshape0*(
_output_shapes
:����������
�
Mtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/Mul_1Mul1batch_normalization_4/cond/batchnorm/mul_1/SwitchOtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/Reshape*
T0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/mul_1*(
_output_shapes
:����������
�
Mtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/Sum_1SumMtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/Mul_1_training/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/mul_1
�
Qtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/Reshape_1ReshapeMtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/Sum_1Otraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/Shape_1*
T0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/mul_1*
Tshape0*
_output_shapes	
:�
�
Itraining/Adam/gradients/batch_normalization_4/cond/batchnorm/sub_grad/NegNegQtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/Reshape_1*
T0*;
_class1
/-loc:@batch_normalization_4/cond/batchnorm/sub*
_output_shapes	
:�
�
Htraining/Adam/gradients/batch_normalization_4/batchnorm/add_1_grad/ShapeShape%batch_normalization_4/batchnorm/mul_1*
_output_shapes
:*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1*
out_type0
�
Jtraining/Adam/gradients/batch_normalization_4/batchnorm/add_1_grad/Shape_1Shape#batch_normalization_4/batchnorm/sub*
_output_shapes
:*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1*
out_type0
�
Xtraining/Adam/gradients/batch_normalization_4/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsHtraining/Adam/gradients/batch_normalization_4/batchnorm/add_1_grad/ShapeJtraining/Adam/gradients/batch_normalization_4/batchnorm/add_1_grad/Shape_1*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1*2
_output_shapes 
:���������:���������
�
Ftraining/Adam/gradients/batch_normalization_4/batchnorm/add_1_grad/SumSumJtraining/Adam/gradients/batch_normalization_4/cond/Switch_1_grad/cond_gradXtraining/Adam/gradients/batch_normalization_4/batchnorm/add_1_grad/BroadcastGradientArgs*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Jtraining/Adam/gradients/batch_normalization_4/batchnorm/add_1_grad/ReshapeReshapeFtraining/Adam/gradients/batch_normalization_4/batchnorm/add_1_grad/SumHtraining/Adam/gradients/batch_normalization_4/batchnorm/add_1_grad/Shape*(
_output_shapes
:����������*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1*
Tshape0
�
Htraining/Adam/gradients/batch_normalization_4/batchnorm/add_1_grad/Sum_1SumJtraining/Adam/gradients/batch_normalization_4/cond/Switch_1_grad/cond_gradZtraining/Adam/gradients/batch_normalization_4/batchnorm/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1*
_output_shapes
:
�
Ltraining/Adam/gradients/batch_normalization_4/batchnorm/add_1_grad/Reshape_1ReshapeHtraining/Adam/gradients/batch_normalization_4/batchnorm/add_1_grad/Sum_1Jtraining/Adam/gradients/batch_normalization_4/batchnorm/add_1_grad/Shape_1*
_output_shapes	
:�*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1*
Tshape0
�
 training/Adam/gradients/Switch_1Switchdense_4/Relu"batch_normalization_4/cond/pred_id*
T0*
_class
loc:@dense_4/Relu*<
_output_shapes*
(:����������:����������
�
"training/Adam/gradients/Identity_1Identity"training/Adam/gradients/Switch_1:1*
T0*
_class
loc:@dense_4/Relu*(
_output_shapes
:����������
�
training/Adam/gradients/Shape_2Shape"training/Adam/gradients/Switch_1:1*
_output_shapes
:*
T0*
_class
loc:@dense_4/Relu*
out_type0
�
%training/Adam/gradients/zeros_1/ConstConst#^training/Adam/gradients/Identity_1*
_class
loc:@dense_4/Relu*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/gradients/zeros_1Filltraining/Adam/gradients/Shape_2%training/Adam/gradients/zeros_1/Const*(
_output_shapes
:����������*
T0*
_class
loc:@dense_4/Relu*

index_type0
�
Xtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_1/Switch_grad/cond_gradMergeOtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/Reshapetraining/Adam/gradients/zeros_1*
T0*
_class
loc:@dense_4/Relu*
N**
_output_shapes
:����������: 
�
Ktraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_2_grad/MulMulItraining/Adam/gradients/batch_normalization_4/cond/batchnorm/sub_grad/Neg(batch_normalization_4/cond/batchnorm/mul*
_output_shapes	
:�*
T0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/mul_2
�
Mtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_2_grad/Mul_1MulItraining/Adam/gradients/batch_normalization_4/cond/batchnorm/sub_grad/Neg5batch_normalization_4/cond/batchnorm/ReadVariableOp_1*
T0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/mul_2*
_output_shapes	
:�
�
Htraining/Adam/gradients/batch_normalization_4/batchnorm/mul_1_grad/ShapeShapedense_4/Relu*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/mul_1*
out_type0*
_output_shapes
:
�
Jtraining/Adam/gradients/batch_normalization_4/batchnorm/mul_1_grad/Shape_1Shape#batch_normalization_4/batchnorm/mul*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/mul_1*
out_type0*
_output_shapes
:
�
Xtraining/Adam/gradients/batch_normalization_4/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsHtraining/Adam/gradients/batch_normalization_4/batchnorm/mul_1_grad/ShapeJtraining/Adam/gradients/batch_normalization_4/batchnorm/mul_1_grad/Shape_1*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/mul_1*2
_output_shapes 
:���������:���������
�
Ftraining/Adam/gradients/batch_normalization_4/batchnorm/mul_1_grad/MulMulJtraining/Adam/gradients/batch_normalization_4/batchnorm/add_1_grad/Reshape#batch_normalization_4/batchnorm/mul*(
_output_shapes
:����������*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/mul_1
�
Ftraining/Adam/gradients/batch_normalization_4/batchnorm/mul_1_grad/SumSumFtraining/Adam/gradients/batch_normalization_4/batchnorm/mul_1_grad/MulXtraining/Adam/gradients/batch_normalization_4/batchnorm/mul_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/mul_1*
_output_shapes
:
�
Jtraining/Adam/gradients/batch_normalization_4/batchnorm/mul_1_grad/ReshapeReshapeFtraining/Adam/gradients/batch_normalization_4/batchnorm/mul_1_grad/SumHtraining/Adam/gradients/batch_normalization_4/batchnorm/mul_1_grad/Shape*(
_output_shapes
:����������*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/mul_1*
Tshape0
�
Htraining/Adam/gradients/batch_normalization_4/batchnorm/mul_1_grad/Mul_1Muldense_4/ReluJtraining/Adam/gradients/batch_normalization_4/batchnorm/add_1_grad/Reshape*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/mul_1*(
_output_shapes
:����������
�
Htraining/Adam/gradients/batch_normalization_4/batchnorm/mul_1_grad/Sum_1SumHtraining/Adam/gradients/batch_normalization_4/batchnorm/mul_1_grad/Mul_1Ztraining/Adam/gradients/batch_normalization_4/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/mul_1*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Ltraining/Adam/gradients/batch_normalization_4/batchnorm/mul_1_grad/Reshape_1ReshapeHtraining/Adam/gradients/batch_normalization_4/batchnorm/mul_1_grad/Sum_1Jtraining/Adam/gradients/batch_normalization_4/batchnorm/mul_1_grad/Shape_1*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/mul_1*
Tshape0*
_output_shapes	
:�
�
Dtraining/Adam/gradients/batch_normalization_4/batchnorm/sub_grad/NegNegLtraining/Adam/gradients/batch_normalization_4/batchnorm/add_1_grad/Reshape_1*
T0*6
_class,
*(loc:@batch_normalization_4/batchnorm/sub*
_output_shapes	
:�
�
 training/Adam/gradients/Switch_2Switchbatch_normalization_4/beta"batch_normalization_4/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_4/beta*
_output_shapes
: : 
�
"training/Adam/gradients/Identity_2Identity"training/Adam/gradients/Switch_2:1*
T0*-
_class#
!loc:@batch_normalization_4/beta*
_output_shapes
: 
�
%training/Adam/gradients/VariableShapeVariableShape"training/Adam/gradients/Switch_2:1#^training/Adam/gradients/Identity_2*-
_class#
!loc:@batch_normalization_4/beta*
out_type0*
_output_shapes
:
�
%training/Adam/gradients/zeros_2/ConstConst#^training/Adam/gradients/Identity_2*-
_class#
!loc:@batch_normalization_4/beta*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/gradients/zeros_2Fill%training/Adam/gradients/VariableShape%training/Adam/gradients/zeros_2/Const*
T0*-
_class#
!loc:@batch_normalization_4/beta*

index_type0*#
_output_shapes
:���������
�
ctraining/Adam/gradients/batch_normalization_4/cond/batchnorm/ReadVariableOp_2/Switch_grad/cond_gradMergeQtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/Reshape_1training/Adam/gradients/zeros_2*
T0*-
_class#
!loc:@batch_normalization_4/beta*
N*%
_output_shapes
:���������: 
�
training/Adam/gradients/AddNAddNQtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/Reshape_1Mtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_2_grad/Mul_1*
N*
_output_shapes	
:�*
T0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/mul_1
�
Itraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_grad/MulMultraining/Adam/gradients/AddN7batch_normalization_4/cond/batchnorm/mul/ReadVariableOp*
T0*;
_class1
/-loc:@batch_normalization_4/cond/batchnorm/mul*
_output_shapes	
:�
�
Ktraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_grad/Mul_1Multraining/Adam/gradients/AddN*batch_normalization_4/cond/batchnorm/Rsqrt*
_output_shapes	
:�*
T0*;
_class1
/-loc:@batch_normalization_4/cond/batchnorm/mul
�
Ftraining/Adam/gradients/batch_normalization_4/batchnorm/mul_2_grad/MulMulDtraining/Adam/gradients/batch_normalization_4/batchnorm/sub_grad/Neg#batch_normalization_4/batchnorm/mul*
_output_shapes	
:�*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/mul_2
�
Htraining/Adam/gradients/batch_normalization_4/batchnorm/mul_2_grad/Mul_1MulDtraining/Adam/gradients/batch_normalization_4/batchnorm/sub_grad/Neg%batch_normalization_4/moments/Squeeze*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/mul_2*
_output_shapes	
:�
�
training/Adam/gradients/AddN_1AddNctraining/Adam/gradients/batch_normalization_4/cond/batchnorm/ReadVariableOp_2/Switch_grad/cond_gradLtraining/Adam/gradients/batch_normalization_4/batchnorm/add_1_grad/Reshape_1*
T0*-
_class#
!loc:@batch_normalization_4/beta*
N*
_output_shapes	
:�
�
Htraining/Adam/gradients/batch_normalization_4/moments/Squeeze_grad/ShapeConst*8
_class.
,*loc:@batch_normalization_4/moments/Squeeze*
valueB"   �   *
dtype0*
_output_shapes
:
�
Jtraining/Adam/gradients/batch_normalization_4/moments/Squeeze_grad/ReshapeReshapeFtraining/Adam/gradients/batch_normalization_4/batchnorm/mul_2_grad/MulHtraining/Adam/gradients/batch_normalization_4/moments/Squeeze_grad/Shape*
T0*8
_class.
,*loc:@batch_normalization_4/moments/Squeeze*
Tshape0*
_output_shapes
:	�
�
training/Adam/gradients/AddN_2AddNLtraining/Adam/gradients/batch_normalization_4/batchnorm/mul_1_grad/Reshape_1Htraining/Adam/gradients/batch_normalization_4/batchnorm/mul_2_grad/Mul_1*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/mul_1*
N*
_output_shapes	
:�
�
Dtraining/Adam/gradients/batch_normalization_4/batchnorm/mul_grad/MulMultraining/Adam/gradients/AddN_22batch_normalization_4/batchnorm/mul/ReadVariableOp*
T0*6
_class,
*(loc:@batch_normalization_4/batchnorm/mul*
_output_shapes	
:�
�
Ftraining/Adam/gradients/batch_normalization_4/batchnorm/mul_grad/Mul_1Multraining/Adam/gradients/AddN_2%batch_normalization_4/batchnorm/Rsqrt*
T0*6
_class,
*(loc:@batch_normalization_4/batchnorm/mul*
_output_shapes	
:�
�
 training/Adam/gradients/Switch_3Switchbatch_normalization_4/gamma"batch_normalization_4/cond/pred_id*
_output_shapes
: : *
T0*.
_class$
" loc:@batch_normalization_4/gamma
�
"training/Adam/gradients/Identity_3Identity"training/Adam/gradients/Switch_3:1*
T0*.
_class$
" loc:@batch_normalization_4/gamma*
_output_shapes
: 
�
'training/Adam/gradients/VariableShape_1VariableShape"training/Adam/gradients/Switch_3:1#^training/Adam/gradients/Identity_3*
_output_shapes
:*.
_class$
" loc:@batch_normalization_4/gamma*
out_type0
�
%training/Adam/gradients/zeros_3/ConstConst#^training/Adam/gradients/Identity_3*
dtype0*
_output_shapes
: *.
_class$
" loc:@batch_normalization_4/gamma*
valueB
 *    
�
training/Adam/gradients/zeros_3Fill'training/Adam/gradients/VariableShape_1%training/Adam/gradients/zeros_3/Const*#
_output_shapes
:���������*
T0*.
_class$
" loc:@batch_normalization_4/gamma*

index_type0
�
etraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul/ReadVariableOp/Switch_grad/cond_gradMergeKtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_grad/Mul_1training/Adam/gradients/zeros_3*
N*%
_output_shapes
:���������: *
T0*.
_class$
" loc:@batch_normalization_4/gamma
�
Ltraining/Adam/gradients/batch_normalization_4/batchnorm/Rsqrt_grad/RsqrtGrad	RsqrtGrad%batch_normalization_4/batchnorm/RsqrtDtraining/Adam/gradients/batch_normalization_4/batchnorm/mul_grad/Mul*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/Rsqrt*
_output_shapes	
:�
�
Ytraining/Adam/gradients/batch_normalization_4/batchnorm/add_grad/BroadcastGradientArgs/s0Const*6
_class,
*(loc:@batch_normalization_4/batchnorm/add*
valueB:�*
dtype0*
_output_shapes
:
�
Ytraining/Adam/gradients/batch_normalization_4/batchnorm/add_grad/BroadcastGradientArgs/s1Const*6
_class,
*(loc:@batch_normalization_4/batchnorm/add*
valueB *
dtype0*
_output_shapes
: 
�
Vtraining/Adam/gradients/batch_normalization_4/batchnorm/add_grad/BroadcastGradientArgsBroadcastGradientArgsYtraining/Adam/gradients/batch_normalization_4/batchnorm/add_grad/BroadcastGradientArgs/s0Ytraining/Adam/gradients/batch_normalization_4/batchnorm/add_grad/BroadcastGradientArgs/s1*
T0*6
_class,
*(loc:@batch_normalization_4/batchnorm/add*2
_output_shapes 
:���������:���������
�
Vtraining/Adam/gradients/batch_normalization_4/batchnorm/add_grad/Sum/reduction_indicesConst*6
_class,
*(loc:@batch_normalization_4/batchnorm/add*
valueB: *
dtype0*
_output_shapes
:
�
Dtraining/Adam/gradients/batch_normalization_4/batchnorm/add_grad/SumSumLtraining/Adam/gradients/batch_normalization_4/batchnorm/Rsqrt_grad/RsqrtGradVtraining/Adam/gradients/batch_normalization_4/batchnorm/add_grad/Sum/reduction_indices*
T0*6
_class,
*(loc:@batch_normalization_4/batchnorm/add*
_output_shapes
: *
	keep_dims( *

Tidx0
�
Ntraining/Adam/gradients/batch_normalization_4/batchnorm/add_grad/Reshape/shapeConst*
dtype0*
_output_shapes
: *6
_class,
*(loc:@batch_normalization_4/batchnorm/add*
valueB 
�
Htraining/Adam/gradients/batch_normalization_4/batchnorm/add_grad/ReshapeReshapeDtraining/Adam/gradients/batch_normalization_4/batchnorm/add_grad/SumNtraining/Adam/gradients/batch_normalization_4/batchnorm/add_grad/Reshape/shape*
_output_shapes
: *
T0*6
_class,
*(loc:@batch_normalization_4/batchnorm/add*
Tshape0
�
training/Adam/gradients/AddN_3AddNetraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul/ReadVariableOp/Switch_grad/cond_gradFtraining/Adam/gradients/batch_normalization_4/batchnorm/mul_grad/Mul_1*
T0*.
_class$
" loc:@batch_normalization_4/gamma*
N*
_output_shapes	
:�
�
Jtraining/Adam/gradients/batch_normalization_4/moments/Squeeze_1_grad/ShapeConst*:
_class0
.,loc:@batch_normalization_4/moments/Squeeze_1*
valueB"   �   *
dtype0*
_output_shapes
:
�
Ltraining/Adam/gradients/batch_normalization_4/moments/Squeeze_1_grad/ReshapeReshapeLtraining/Adam/gradients/batch_normalization_4/batchnorm/Rsqrt_grad/RsqrtGradJtraining/Adam/gradients/batch_normalization_4/moments/Squeeze_1_grad/Shape*
T0*:
_class0
.,loc:@batch_normalization_4/moments/Squeeze_1*
Tshape0*
_output_shapes
:	�
�
Itraining/Adam/gradients/batch_normalization_4/moments/variance_grad/ShapeShape/batch_normalization_4/moments/SquaredDifference*
T0*9
_class/
-+loc:@batch_normalization_4/moments/variance*
out_type0*
_output_shapes
:
�
Htraining/Adam/gradients/batch_normalization_4/moments/variance_grad/SizeConst*9
_class/
-+loc:@batch_normalization_4/moments/variance*
value	B :*
dtype0*
_output_shapes
: 
�
Gtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/addAddV28batch_normalization_4/moments/variance/reduction_indicesHtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/Size*
T0*9
_class/
-+loc:@batch_normalization_4/moments/variance*
_output_shapes
:
�
Gtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/modFloorModGtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/addHtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/Size*
T0*9
_class/
-+loc:@batch_normalization_4/moments/variance*
_output_shapes
:
�
Ktraining/Adam/gradients/batch_normalization_4/moments/variance_grad/Shape_1Const*9
_class/
-+loc:@batch_normalization_4/moments/variance*
valueB:*
dtype0*
_output_shapes
:
�
Otraining/Adam/gradients/batch_normalization_4/moments/variance_grad/range/startConst*9
_class/
-+loc:@batch_normalization_4/moments/variance*
value	B : *
dtype0*
_output_shapes
: 
�
Otraining/Adam/gradients/batch_normalization_4/moments/variance_grad/range/deltaConst*9
_class/
-+loc:@batch_normalization_4/moments/variance*
value	B :*
dtype0*
_output_shapes
: 
�
Itraining/Adam/gradients/batch_normalization_4/moments/variance_grad/rangeRangeOtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/range/startHtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/SizeOtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/range/delta*
_output_shapes
:*

Tidx0*9
_class/
-+loc:@batch_normalization_4/moments/variance
�
Ntraining/Adam/gradients/batch_normalization_4/moments/variance_grad/Fill/valueConst*9
_class/
-+loc:@batch_normalization_4/moments/variance*
value	B :*
dtype0*
_output_shapes
: 
�
Htraining/Adam/gradients/batch_normalization_4/moments/variance_grad/FillFillKtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/Shape_1Ntraining/Adam/gradients/batch_normalization_4/moments/variance_grad/Fill/value*
_output_shapes
:*
T0*9
_class/
-+loc:@batch_normalization_4/moments/variance*

index_type0
�
Qtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/DynamicStitchDynamicStitchItraining/Adam/gradients/batch_normalization_4/moments/variance_grad/rangeGtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/modItraining/Adam/gradients/batch_normalization_4/moments/variance_grad/ShapeHtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/Fill*
N*
_output_shapes
:*
T0*9
_class/
-+loc:@batch_normalization_4/moments/variance
�
Mtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/Maximum/yConst*9
_class/
-+loc:@batch_normalization_4/moments/variance*
value	B :*
dtype0*
_output_shapes
: 
�
Ktraining/Adam/gradients/batch_normalization_4/moments/variance_grad/MaximumMaximumQtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/DynamicStitchMtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/Maximum/y*
T0*9
_class/
-+loc:@batch_normalization_4/moments/variance*
_output_shapes
:
�
Ltraining/Adam/gradients/batch_normalization_4/moments/variance_grad/floordivFloorDivItraining/Adam/gradients/batch_normalization_4/moments/variance_grad/ShapeKtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/Maximum*
T0*9
_class/
-+loc:@batch_normalization_4/moments/variance*
_output_shapes
:
�
Ktraining/Adam/gradients/batch_normalization_4/moments/variance_grad/ReshapeReshapeLtraining/Adam/gradients/batch_normalization_4/moments/Squeeze_1_grad/ReshapeQtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/DynamicStitch*
T0*9
_class/
-+loc:@batch_normalization_4/moments/variance*
Tshape0*0
_output_shapes
:������������������
�
Htraining/Adam/gradients/batch_normalization_4/moments/variance_grad/TileTileKtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/ReshapeLtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/floordiv*0
_output_shapes
:������������������*

Tmultiples0*
T0*9
_class/
-+loc:@batch_normalization_4/moments/variance
�
Ktraining/Adam/gradients/batch_normalization_4/moments/variance_grad/Shape_2Shape/batch_normalization_4/moments/SquaredDifference*
T0*9
_class/
-+loc:@batch_normalization_4/moments/variance*
out_type0*
_output_shapes
:
�
Ktraining/Adam/gradients/batch_normalization_4/moments/variance_grad/Shape_3Const*9
_class/
-+loc:@batch_normalization_4/moments/variance*
valueB"   �   *
dtype0*
_output_shapes
:
�
Itraining/Adam/gradients/batch_normalization_4/moments/variance_grad/ConstConst*
dtype0*
_output_shapes
:*9
_class/
-+loc:@batch_normalization_4/moments/variance*
valueB: 
�
Htraining/Adam/gradients/batch_normalization_4/moments/variance_grad/ProdProdKtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/Shape_2Itraining/Adam/gradients/batch_normalization_4/moments/variance_grad/Const*
T0*9
_class/
-+loc:@batch_normalization_4/moments/variance*
_output_shapes
: *
	keep_dims( *

Tidx0
�
Ktraining/Adam/gradients/batch_normalization_4/moments/variance_grad/Const_1Const*9
_class/
-+loc:@batch_normalization_4/moments/variance*
valueB: *
dtype0*
_output_shapes
:
�
Jtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/Prod_1ProdKtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/Shape_3Ktraining/Adam/gradients/batch_normalization_4/moments/variance_grad/Const_1*
T0*9
_class/
-+loc:@batch_normalization_4/moments/variance*
_output_shapes
: *
	keep_dims( *

Tidx0
�
Otraining/Adam/gradients/batch_normalization_4/moments/variance_grad/Maximum_1/yConst*
dtype0*
_output_shapes
: *9
_class/
-+loc:@batch_normalization_4/moments/variance*
value	B :
�
Mtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/Maximum_1MaximumJtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/Prod_1Otraining/Adam/gradients/batch_normalization_4/moments/variance_grad/Maximum_1/y*
_output_shapes
: *
T0*9
_class/
-+loc:@batch_normalization_4/moments/variance
�
Ntraining/Adam/gradients/batch_normalization_4/moments/variance_grad/floordiv_1FloorDivHtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/ProdMtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/Maximum_1*
_output_shapes
: *
T0*9
_class/
-+loc:@batch_normalization_4/moments/variance
�
Htraining/Adam/gradients/batch_normalization_4/moments/variance_grad/CastCastNtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/floordiv_1*

SrcT0*9
_class/
-+loc:@batch_normalization_4/moments/variance*
Truncate( *

DstT0*
_output_shapes
: 
�
Ktraining/Adam/gradients/batch_normalization_4/moments/variance_grad/truedivRealDivHtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/TileHtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/Cast*
T0*9
_class/
-+loc:@batch_normalization_4/moments/variance*(
_output_shapes
:����������
�
Straining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/scalarConstL^training/Adam/gradients/batch_normalization_4/moments/variance_grad/truediv*
dtype0*
_output_shapes
: *B
_class8
64loc:@batch_normalization_4/moments/SquaredDifference*
valueB
 *   @
�
Ptraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/MulMulStraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/scalarKtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/truediv*(
_output_shapes
:����������*
T0*B
_class8
64loc:@batch_normalization_4/moments/SquaredDifference
�
Ptraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/subSubdense_4/Relu*batch_normalization_4/moments/StopGradientL^training/Adam/gradients/batch_normalization_4/moments/variance_grad/truediv*
T0*B
_class8
64loc:@batch_normalization_4/moments/SquaredDifference*(
_output_shapes
:����������
�
Rtraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/mul_1MulPtraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/MulPtraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/sub*(
_output_shapes
:����������*
T0*B
_class8
64loc:@batch_normalization_4/moments/SquaredDifference
�
Rtraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/ShapeShapedense_4/Relu*
T0*B
_class8
64loc:@batch_normalization_4/moments/SquaredDifference*
out_type0*
_output_shapes
:
�
Ttraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/Shape_1Shape*batch_normalization_4/moments/StopGradient*
_output_shapes
:*
T0*B
_class8
64loc:@batch_normalization_4/moments/SquaredDifference*
out_type0
�
btraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgsRtraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/ShapeTtraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/Shape_1*
T0*B
_class8
64loc:@batch_normalization_4/moments/SquaredDifference*2
_output_shapes 
:���������:���������
�
Ptraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/SumSumRtraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/mul_1btraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/BroadcastGradientArgs*
T0*B
_class8
64loc:@batch_normalization_4/moments/SquaredDifference*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Ttraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/ReshapeReshapePtraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/SumRtraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/Shape*
T0*B
_class8
64loc:@batch_normalization_4/moments/SquaredDifference*
Tshape0*(
_output_shapes
:����������
�
Rtraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/Sum_1SumRtraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/mul_1dtraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*B
_class8
64loc:@batch_normalization_4/moments/SquaredDifference*
_output_shapes
:
�
Vtraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/Reshape_1ReshapeRtraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/Sum_1Ttraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/Shape_1*
T0*B
_class8
64loc:@batch_normalization_4/moments/SquaredDifference*
Tshape0*
_output_shapes
:	�
�
Ptraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/NegNegVtraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/Reshape_1*
_output_shapes
:	�*
T0*B
_class8
64loc:@batch_normalization_4/moments/SquaredDifference
�
Etraining/Adam/gradients/batch_normalization_4/moments/mean_grad/ShapeShapedense_4/Relu*
T0*5
_class+
)'loc:@batch_normalization_4/moments/mean*
out_type0*
_output_shapes
:
�
Dtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/SizeConst*5
_class+
)'loc:@batch_normalization_4/moments/mean*
value	B :*
dtype0*
_output_shapes
: 
�
Ctraining/Adam/gradients/batch_normalization_4/moments/mean_grad/addAddV24batch_normalization_4/moments/mean/reduction_indicesDtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Size*
_output_shapes
:*
T0*5
_class+
)'loc:@batch_normalization_4/moments/mean
�
Ctraining/Adam/gradients/batch_normalization_4/moments/mean_grad/modFloorModCtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/addDtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Size*
_output_shapes
:*
T0*5
_class+
)'loc:@batch_normalization_4/moments/mean
�
Gtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Shape_1Const*5
_class+
)'loc:@batch_normalization_4/moments/mean*
valueB:*
dtype0*
_output_shapes
:
�
Ktraining/Adam/gradients/batch_normalization_4/moments/mean_grad/range/startConst*
dtype0*
_output_shapes
: *5
_class+
)'loc:@batch_normalization_4/moments/mean*
value	B : 
�
Ktraining/Adam/gradients/batch_normalization_4/moments/mean_grad/range/deltaConst*
dtype0*
_output_shapes
: *5
_class+
)'loc:@batch_normalization_4/moments/mean*
value	B :
�
Etraining/Adam/gradients/batch_normalization_4/moments/mean_grad/rangeRangeKtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/range/startDtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/SizeKtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/range/delta*5
_class+
)'loc:@batch_normalization_4/moments/mean*
_output_shapes
:*

Tidx0
�
Jtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Fill/valueConst*5
_class+
)'loc:@batch_normalization_4/moments/mean*
value	B :*
dtype0*
_output_shapes
: 
�
Dtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/FillFillGtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Shape_1Jtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Fill/value*
T0*5
_class+
)'loc:@batch_normalization_4/moments/mean*

index_type0*
_output_shapes
:
�
Mtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/DynamicStitchDynamicStitchEtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/rangeCtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/modEtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/ShapeDtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Fill*
T0*5
_class+
)'loc:@batch_normalization_4/moments/mean*
N*
_output_shapes
:
�
Itraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Maximum/yConst*5
_class+
)'loc:@batch_normalization_4/moments/mean*
value	B :*
dtype0*
_output_shapes
: 
�
Gtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/MaximumMaximumMtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/DynamicStitchItraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Maximum/y*
T0*5
_class+
)'loc:@batch_normalization_4/moments/mean*
_output_shapes
:
�
Htraining/Adam/gradients/batch_normalization_4/moments/mean_grad/floordivFloorDivEtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/ShapeGtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Maximum*
T0*5
_class+
)'loc:@batch_normalization_4/moments/mean*
_output_shapes
:
�
Gtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/ReshapeReshapeJtraining/Adam/gradients/batch_normalization_4/moments/Squeeze_grad/ReshapeMtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/DynamicStitch*0
_output_shapes
:������������������*
T0*5
_class+
)'loc:@batch_normalization_4/moments/mean*
Tshape0
�
Dtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/TileTileGtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/ReshapeHtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/floordiv*
T0*5
_class+
)'loc:@batch_normalization_4/moments/mean*0
_output_shapes
:������������������*

Tmultiples0
�
Gtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Shape_2Shapedense_4/Relu*
_output_shapes
:*
T0*5
_class+
)'loc:@batch_normalization_4/moments/mean*
out_type0
�
Gtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Shape_3Const*
dtype0*
_output_shapes
:*5
_class+
)'loc:@batch_normalization_4/moments/mean*
valueB"   �   
�
Etraining/Adam/gradients/batch_normalization_4/moments/mean_grad/ConstConst*5
_class+
)'loc:@batch_normalization_4/moments/mean*
valueB: *
dtype0*
_output_shapes
:
�
Dtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/ProdProdGtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Shape_2Etraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Const*
T0*5
_class+
)'loc:@batch_normalization_4/moments/mean*
_output_shapes
: *
	keep_dims( *

Tidx0
�
Gtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Const_1Const*5
_class+
)'loc:@batch_normalization_4/moments/mean*
valueB: *
dtype0*
_output_shapes
:
�
Ftraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Prod_1ProdGtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Shape_3Gtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0*5
_class+
)'loc:@batch_normalization_4/moments/mean
�
Ktraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Maximum_1/yConst*5
_class+
)'loc:@batch_normalization_4/moments/mean*
value	B :*
dtype0*
_output_shapes
: 
�
Itraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Maximum_1MaximumFtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Prod_1Ktraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Maximum_1/y*
T0*5
_class+
)'loc:@batch_normalization_4/moments/mean*
_output_shapes
: 
�
Jtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/floordiv_1FloorDivDtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/ProdItraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Maximum_1*
T0*5
_class+
)'loc:@batch_normalization_4/moments/mean*
_output_shapes
: 
�
Dtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/CastCastJtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/floordiv_1*

SrcT0*5
_class+
)'loc:@batch_normalization_4/moments/mean*
Truncate( *

DstT0*
_output_shapes
: 
�
Gtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/truedivRealDivDtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/TileDtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Cast*
T0*5
_class+
)'loc:@batch_normalization_4/moments/mean*(
_output_shapes
:����������
�
training/Adam/gradients/AddN_4AddNXtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_1/Switch_grad/cond_gradJtraining/Adam/gradients/batch_normalization_4/batchnorm/mul_1_grad/ReshapeTtraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/ReshapeGtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/truediv*
T0*
_class
loc:@dense_4/Relu*
N*(
_output_shapes
:����������
�
2training/Adam/gradients/dense_4/Relu_grad/ReluGradReluGradtraining/Adam/gradients/AddN_4dense_4/Relu*
T0*
_class
loc:@dense_4/Relu*(
_output_shapes
:����������
�
8training/Adam/gradients/dense_4/BiasAdd_grad/BiasAddGradBiasAddGrad2training/Adam/gradients/dense_4/Relu_grad/ReluGrad*
T0*"
_class
loc:@dense_4/BiasAdd*
data_formatNHWC*
_output_shapes	
:�
�
2training/Adam/gradients/dense_4/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_4/Relu_grad/ReluGraddense_4/MatMul/ReadVariableOp*
T0*!
_class
loc:@dense_4/MatMul*
transpose_a( *(
_output_shapes
:����������*
transpose_b(
�
4training/Adam/gradients/dense_4/MatMul_grad/MatMul_1MatMuldropout_2/cond/Merge2training/Adam/gradients/dense_4/Relu_grad/ReluGrad*
transpose_a(* 
_output_shapes
:
��*
transpose_b( *
T0*!
_class
loc:@dense_4/MatMul
�
;training/Adam/gradients/dropout_2/cond/Merge_grad/cond_gradSwitch2training/Adam/gradients/dense_4/MatMul_grad/MatMuldropout_2/cond/pred_id*
T0*!
_class
loc:@dense_4/MatMul*<
_output_shapes*
(:����������:����������
�
 training/Adam/gradients/Switch_4Switch batch_normalization_3/cond/Mergedropout_2/cond/pred_id*<
_output_shapes*
(:����������:����������*
T0*3
_class)
'%loc:@batch_normalization_3/cond/Merge
�
"training/Adam/gradients/Identity_4Identity"training/Adam/gradients/Switch_4:1*
T0*3
_class)
'%loc:@batch_normalization_3/cond/Merge*(
_output_shapes
:����������
�
training/Adam/gradients/Shape_3Shape"training/Adam/gradients/Switch_4:1*
T0*3
_class)
'%loc:@batch_normalization_3/cond/Merge*
out_type0*
_output_shapes
:
�
%training/Adam/gradients/zeros_4/ConstConst#^training/Adam/gradients/Identity_4*3
_class)
'%loc:@batch_normalization_3/cond/Merge*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/gradients/zeros_4Filltraining/Adam/gradients/Shape_3%training/Adam/gradients/zeros_4/Const*
T0*3
_class)
'%loc:@batch_normalization_3/cond/Merge*

index_type0*(
_output_shapes
:����������
�
>training/Adam/gradients/dropout_2/cond/Switch_1_grad/cond_gradMerge;training/Adam/gradients/dropout_2/cond/Merge_grad/cond_gradtraining/Adam/gradients/zeros_4*
N**
_output_shapes
:����������: *
T0*3
_class)
'%loc:@batch_normalization_3/cond/Merge
�
?training/Adam/gradients/dropout_2/cond/dropout/mul_1_grad/ShapeShapedropout_2/cond/dropout/mul*
T0*/
_class%
#!loc:@dropout_2/cond/dropout/mul_1*
out_type0*
_output_shapes
:
�
Atraining/Adam/gradients/dropout_2/cond/dropout/mul_1_grad/Shape_1Shapedropout_2/cond/dropout/Cast*
T0*/
_class%
#!loc:@dropout_2/cond/dropout/mul_1*
out_type0*
_output_shapes
:
�
Otraining/Adam/gradients/dropout_2/cond/dropout/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs?training/Adam/gradients/dropout_2/cond/dropout/mul_1_grad/ShapeAtraining/Adam/gradients/dropout_2/cond/dropout/mul_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0*/
_class%
#!loc:@dropout_2/cond/dropout/mul_1
�
=training/Adam/gradients/dropout_2/cond/dropout/mul_1_grad/MulMul=training/Adam/gradients/dropout_2/cond/Merge_grad/cond_grad:1dropout_2/cond/dropout/Cast*(
_output_shapes
:����������*
T0*/
_class%
#!loc:@dropout_2/cond/dropout/mul_1
�
=training/Adam/gradients/dropout_2/cond/dropout/mul_1_grad/SumSum=training/Adam/gradients/dropout_2/cond/dropout/mul_1_grad/MulOtraining/Adam/gradients/dropout_2/cond/dropout/mul_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*/
_class%
#!loc:@dropout_2/cond/dropout/mul_1*
_output_shapes
:
�
Atraining/Adam/gradients/dropout_2/cond/dropout/mul_1_grad/ReshapeReshape=training/Adam/gradients/dropout_2/cond/dropout/mul_1_grad/Sum?training/Adam/gradients/dropout_2/cond/dropout/mul_1_grad/Shape*
T0*/
_class%
#!loc:@dropout_2/cond/dropout/mul_1*
Tshape0*(
_output_shapes
:����������
�
?training/Adam/gradients/dropout_2/cond/dropout/mul_1_grad/Mul_1Muldropout_2/cond/dropout/mul=training/Adam/gradients/dropout_2/cond/Merge_grad/cond_grad:1*
T0*/
_class%
#!loc:@dropout_2/cond/dropout/mul_1*(
_output_shapes
:����������
�
?training/Adam/gradients/dropout_2/cond/dropout/mul_1_grad/Sum_1Sum?training/Adam/gradients/dropout_2/cond/dropout/mul_1_grad/Mul_1Qtraining/Adam/gradients/dropout_2/cond/dropout/mul_1_grad/BroadcastGradientArgs:1*
T0*/
_class%
#!loc:@dropout_2/cond/dropout/mul_1*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Ctraining/Adam/gradients/dropout_2/cond/dropout/mul_1_grad/Reshape_1Reshape?training/Adam/gradients/dropout_2/cond/dropout/mul_1_grad/Sum_1Atraining/Adam/gradients/dropout_2/cond/dropout/mul_1_grad/Shape_1*
T0*/
_class%
#!loc:@dropout_2/cond/dropout/mul_1*
Tshape0*(
_output_shapes
:����������
�
=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/ShapeShape%dropout_2/cond/dropout/Shape/Switch:1*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul*
out_type0*
_output_shapes
:
�
?training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Shape_1Shapedropout_2/cond/dropout/truediv*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul*
out_type0*
_output_shapes
: 
�
Mtraining/Adam/gradients/dropout_2/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Shape?training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul
�
;training/Adam/gradients/dropout_2/cond/dropout/mul_grad/MulMulAtraining/Adam/gradients/dropout_2/cond/dropout/mul_1_grad/Reshapedropout_2/cond/dropout/truediv*(
_output_shapes
:����������*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul
�
;training/Adam/gradients/dropout_2/cond/dropout/mul_grad/SumSum;training/Adam/gradients/dropout_2/cond/dropout/mul_grad/MulMtraining/Adam/gradients/dropout_2/cond/dropout/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul
�
?training/Adam/gradients/dropout_2/cond/dropout/mul_grad/ReshapeReshape;training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Sum=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Shape*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul*
Tshape0*(
_output_shapes
:����������
�
=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Mul_1Mul%dropout_2/cond/dropout/Shape/Switch:1Atraining/Adam/gradients/dropout_2/cond/dropout/mul_1_grad/Reshape*(
_output_shapes
:����������*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul
�
=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Sum_1Sum=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Mul_1Otraining/Adam/gradients/dropout_2/cond/dropout/mul_grad/BroadcastGradientArgs:1*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Atraining/Adam/gradients/dropout_2/cond/dropout/mul_grad/Reshape_1Reshape=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Sum_1?training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Shape_1*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul*
Tshape0*
_output_shapes
: 
�
 training/Adam/gradients/Switch_5Switch batch_normalization_3/cond/Mergedropout_2/cond/pred_id*
T0*3
_class)
'%loc:@batch_normalization_3/cond/Merge*<
_output_shapes*
(:����������:����������
�
"training/Adam/gradients/Identity_5Identity training/Adam/gradients/Switch_5*(
_output_shapes
:����������*
T0*3
_class)
'%loc:@batch_normalization_3/cond/Merge
�
training/Adam/gradients/Shape_4Shape training/Adam/gradients/Switch_5*
T0*3
_class)
'%loc:@batch_normalization_3/cond/Merge*
out_type0*
_output_shapes
:
�
%training/Adam/gradients/zeros_5/ConstConst#^training/Adam/gradients/Identity_5*
dtype0*
_output_shapes
: *3
_class)
'%loc:@batch_normalization_3/cond/Merge*
valueB
 *    
�
training/Adam/gradients/zeros_5Filltraining/Adam/gradients/Shape_4%training/Adam/gradients/zeros_5/Const*
T0*3
_class)
'%loc:@batch_normalization_3/cond/Merge*

index_type0*(
_output_shapes
:����������
�
Jtraining/Adam/gradients/dropout_2/cond/dropout/Shape/Switch_grad/cond_gradMergetraining/Adam/gradients/zeros_5?training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Reshape*
T0*3
_class)
'%loc:@batch_normalization_3/cond/Merge*
N**
_output_shapes
:����������: 
�
training/Adam/gradients/AddN_5AddN>training/Adam/gradients/dropout_2/cond/Switch_1_grad/cond_gradJtraining/Adam/gradients/dropout_2/cond/dropout/Shape/Switch_grad/cond_grad*
T0*3
_class)
'%loc:@batch_normalization_3/cond/Merge*
N*(
_output_shapes
:����������
�
Gtraining/Adam/gradients/batch_normalization_3/cond/Merge_grad/cond_gradSwitchtraining/Adam/gradients/AddN_5"batch_normalization_3/cond/pred_id*<
_output_shapes*
(:����������:����������*
T0*3
_class)
'%loc:@batch_normalization_3/cond/Merge
�
Mtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/ShapeShape*batch_normalization_3/cond/batchnorm/mul_1*
_output_shapes
:*
T0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/add_1*
out_type0
�
Otraining/Adam/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/Shape_1Shape(batch_normalization_3/cond/batchnorm/sub*
T0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/add_1*
out_type0*
_output_shapes
:
�
]training/Adam/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsMtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/ShapeOtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/add_1
�
Ktraining/Adam/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/SumSumGtraining/Adam/gradients/batch_normalization_3/cond/Merge_grad/cond_grad]training/Adam/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/add_1*
_output_shapes
:
�
Otraining/Adam/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/ReshapeReshapeKtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/SumMtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/Shape*(
_output_shapes
:����������*
T0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/add_1*
Tshape0
�
Mtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/Sum_1SumGtraining/Adam/gradients/batch_normalization_3/cond/Merge_grad/cond_grad_training/Adam/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/add_1*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Qtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/Reshape_1ReshapeMtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/Sum_1Otraining/Adam/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/Shape_1*
_output_shapes	
:�*
T0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/add_1*
Tshape0
�
 training/Adam/gradients/Switch_6Switch%batch_normalization_3/batchnorm/add_1"batch_normalization_3/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1*<
_output_shapes*
(:����������:����������
�
"training/Adam/gradients/Identity_6Identity training/Adam/gradients/Switch_6*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1*(
_output_shapes
:����������
�
training/Adam/gradients/Shape_5Shape training/Adam/gradients/Switch_6*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1*
out_type0*
_output_shapes
:
�
%training/Adam/gradients/zeros_6/ConstConst#^training/Adam/gradients/Identity_6*
dtype0*
_output_shapes
: *8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1*
valueB
 *    
�
training/Adam/gradients/zeros_6Filltraining/Adam/gradients/Shape_5%training/Adam/gradients/zeros_6/Const*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1*

index_type0*(
_output_shapes
:����������
�
Jtraining/Adam/gradients/batch_normalization_3/cond/Switch_1_grad/cond_gradMergetraining/Adam/gradients/zeros_6Itraining/Adam/gradients/batch_normalization_3/cond/Merge_grad/cond_grad:1*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1*
N**
_output_shapes
:����������: 
�
Mtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/ShapeShape1batch_normalization_3/cond/batchnorm/mul_1/Switch*
T0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/mul_1*
out_type0*
_output_shapes
:
�
Otraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/Shape_1Shape(batch_normalization_3/cond/batchnorm/mul*
T0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/mul_1*
out_type0*
_output_shapes
:
�
]training/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsMtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/ShapeOtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/Shape_1*
T0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/mul_1*2
_output_shapes 
:���������:���������
�
Ktraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/MulMulOtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/Reshape(batch_normalization_3/cond/batchnorm/mul*
T0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/mul_1*(
_output_shapes
:����������
�
Ktraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/SumSumKtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/Mul]training/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/BroadcastGradientArgs*
T0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/mul_1*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Otraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/ReshapeReshapeKtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/SumMtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/Shape*
T0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/mul_1*
Tshape0*(
_output_shapes
:����������
�
Mtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/Mul_1Mul1batch_normalization_3/cond/batchnorm/mul_1/SwitchOtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/Reshape*
T0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/mul_1*(
_output_shapes
:����������
�
Mtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/Sum_1SumMtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/Mul_1_training/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/mul_1
�
Qtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/Reshape_1ReshapeMtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/Sum_1Otraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/Shape_1*
T0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/mul_1*
Tshape0*
_output_shapes	
:�
�
Itraining/Adam/gradients/batch_normalization_3/cond/batchnorm/sub_grad/NegNegQtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/Reshape_1*
T0*;
_class1
/-loc:@batch_normalization_3/cond/batchnorm/sub*
_output_shapes	
:�
�
Htraining/Adam/gradients/batch_normalization_3/batchnorm/add_1_grad/ShapeShape%batch_normalization_3/batchnorm/mul_1*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1*
out_type0*
_output_shapes
:
�
Jtraining/Adam/gradients/batch_normalization_3/batchnorm/add_1_grad/Shape_1Shape#batch_normalization_3/batchnorm/sub*
_output_shapes
:*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1*
out_type0
�
Xtraining/Adam/gradients/batch_normalization_3/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsHtraining/Adam/gradients/batch_normalization_3/batchnorm/add_1_grad/ShapeJtraining/Adam/gradients/batch_normalization_3/batchnorm/add_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1
�
Ftraining/Adam/gradients/batch_normalization_3/batchnorm/add_1_grad/SumSumJtraining/Adam/gradients/batch_normalization_3/cond/Switch_1_grad/cond_gradXtraining/Adam/gradients/batch_normalization_3/batchnorm/add_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1*
_output_shapes
:
�
Jtraining/Adam/gradients/batch_normalization_3/batchnorm/add_1_grad/ReshapeReshapeFtraining/Adam/gradients/batch_normalization_3/batchnorm/add_1_grad/SumHtraining/Adam/gradients/batch_normalization_3/batchnorm/add_1_grad/Shape*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1*
Tshape0*(
_output_shapes
:����������
�
Htraining/Adam/gradients/batch_normalization_3/batchnorm/add_1_grad/Sum_1SumJtraining/Adam/gradients/batch_normalization_3/cond/Switch_1_grad/cond_gradZtraining/Adam/gradients/batch_normalization_3/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Ltraining/Adam/gradients/batch_normalization_3/batchnorm/add_1_grad/Reshape_1ReshapeHtraining/Adam/gradients/batch_normalization_3/batchnorm/add_1_grad/Sum_1Jtraining/Adam/gradients/batch_normalization_3/batchnorm/add_1_grad/Shape_1*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1*
Tshape0*
_output_shapes	
:�
�
 training/Adam/gradients/Switch_7Switchdense_3/Relu"batch_normalization_3/cond/pred_id*
T0*
_class
loc:@dense_3/Relu*<
_output_shapes*
(:����������:����������
�
"training/Adam/gradients/Identity_7Identity"training/Adam/gradients/Switch_7:1*
T0*
_class
loc:@dense_3/Relu*(
_output_shapes
:����������
�
training/Adam/gradients/Shape_6Shape"training/Adam/gradients/Switch_7:1*
T0*
_class
loc:@dense_3/Relu*
out_type0*
_output_shapes
:
�
%training/Adam/gradients/zeros_7/ConstConst#^training/Adam/gradients/Identity_7*
_class
loc:@dense_3/Relu*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/gradients/zeros_7Filltraining/Adam/gradients/Shape_6%training/Adam/gradients/zeros_7/Const*
T0*
_class
loc:@dense_3/Relu*

index_type0*(
_output_shapes
:����������
�
Xtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_1/Switch_grad/cond_gradMergeOtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/Reshapetraining/Adam/gradients/zeros_7*
T0*
_class
loc:@dense_3/Relu*
N**
_output_shapes
:����������: 
�
Ktraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_2_grad/MulMulItraining/Adam/gradients/batch_normalization_3/cond/batchnorm/sub_grad/Neg(batch_normalization_3/cond/batchnorm/mul*
T0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/mul_2*
_output_shapes	
:�
�
Mtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_2_grad/Mul_1MulItraining/Adam/gradients/batch_normalization_3/cond/batchnorm/sub_grad/Neg5batch_normalization_3/cond/batchnorm/ReadVariableOp_1*
_output_shapes	
:�*
T0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/mul_2
�
Htraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/ShapeShapedense_3/Relu*
_output_shapes
:*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/mul_1*
out_type0
�
Jtraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/Shape_1Shape#batch_normalization_3/batchnorm/mul*
_output_shapes
:*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/mul_1*
out_type0
�
Xtraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsHtraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/ShapeJtraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/Shape_1*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/mul_1*2
_output_shapes 
:���������:���������
�
Ftraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/MulMulJtraining/Adam/gradients/batch_normalization_3/batchnorm/add_1_grad/Reshape#batch_normalization_3/batchnorm/mul*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/mul_1*(
_output_shapes
:����������
�
Ftraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/SumSumFtraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/MulXtraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/mul_1*
_output_shapes
:
�
Jtraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/ReshapeReshapeFtraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/SumHtraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/Shape*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/mul_1*
Tshape0*(
_output_shapes
:����������
�
Htraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/Mul_1Muldense_3/ReluJtraining/Adam/gradients/batch_normalization_3/batchnorm/add_1_grad/Reshape*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/mul_1*(
_output_shapes
:����������
�
Htraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/Sum_1SumHtraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/Mul_1Ztraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/mul_1*
_output_shapes
:
�
Ltraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/Reshape_1ReshapeHtraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/Sum_1Jtraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/Shape_1*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/mul_1*
Tshape0*
_output_shapes	
:�
�
Dtraining/Adam/gradients/batch_normalization_3/batchnorm/sub_grad/NegNegLtraining/Adam/gradients/batch_normalization_3/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:�*
T0*6
_class,
*(loc:@batch_normalization_3/batchnorm/sub
�
 training/Adam/gradients/Switch_8Switchbatch_normalization_3/beta"batch_normalization_3/cond/pred_id*
_output_shapes
: : *
T0*-
_class#
!loc:@batch_normalization_3/beta
�
"training/Adam/gradients/Identity_8Identity"training/Adam/gradients/Switch_8:1*
T0*-
_class#
!loc:@batch_normalization_3/beta*
_output_shapes
: 
�
'training/Adam/gradients/VariableShape_2VariableShape"training/Adam/gradients/Switch_8:1#^training/Adam/gradients/Identity_8*-
_class#
!loc:@batch_normalization_3/beta*
out_type0*
_output_shapes
:
�
%training/Adam/gradients/zeros_8/ConstConst#^training/Adam/gradients/Identity_8*
dtype0*
_output_shapes
: *-
_class#
!loc:@batch_normalization_3/beta*
valueB
 *    
�
training/Adam/gradients/zeros_8Fill'training/Adam/gradients/VariableShape_2%training/Adam/gradients/zeros_8/Const*
T0*-
_class#
!loc:@batch_normalization_3/beta*

index_type0*#
_output_shapes
:���������
�
ctraining/Adam/gradients/batch_normalization_3/cond/batchnorm/ReadVariableOp_2/Switch_grad/cond_gradMergeQtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/Reshape_1training/Adam/gradients/zeros_8*
N*%
_output_shapes
:���������: *
T0*-
_class#
!loc:@batch_normalization_3/beta
�
training/Adam/gradients/AddN_6AddNQtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/Reshape_1Mtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_2_grad/Mul_1*
T0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/mul_1*
N*
_output_shapes	
:�
�
Itraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_grad/MulMultraining/Adam/gradients/AddN_67batch_normalization_3/cond/batchnorm/mul/ReadVariableOp*
T0*;
_class1
/-loc:@batch_normalization_3/cond/batchnorm/mul*
_output_shapes	
:�
�
Ktraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_grad/Mul_1Multraining/Adam/gradients/AddN_6*batch_normalization_3/cond/batchnorm/Rsqrt*
T0*;
_class1
/-loc:@batch_normalization_3/cond/batchnorm/mul*
_output_shapes	
:�
�
Ftraining/Adam/gradients/batch_normalization_3/batchnorm/mul_2_grad/MulMulDtraining/Adam/gradients/batch_normalization_3/batchnorm/sub_grad/Neg#batch_normalization_3/batchnorm/mul*
_output_shapes	
:�*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/mul_2
�
Htraining/Adam/gradients/batch_normalization_3/batchnorm/mul_2_grad/Mul_1MulDtraining/Adam/gradients/batch_normalization_3/batchnorm/sub_grad/Neg%batch_normalization_3/moments/Squeeze*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/mul_2*
_output_shapes	
:�
�
training/Adam/gradients/AddN_7AddNctraining/Adam/gradients/batch_normalization_3/cond/batchnorm/ReadVariableOp_2/Switch_grad/cond_gradLtraining/Adam/gradients/batch_normalization_3/batchnorm/add_1_grad/Reshape_1*
N*
_output_shapes	
:�*
T0*-
_class#
!loc:@batch_normalization_3/beta
�
Htraining/Adam/gradients/batch_normalization_3/moments/Squeeze_grad/ShapeConst*8
_class.
,*loc:@batch_normalization_3/moments/Squeeze*
valueB"   �   *
dtype0*
_output_shapes
:
�
Jtraining/Adam/gradients/batch_normalization_3/moments/Squeeze_grad/ReshapeReshapeFtraining/Adam/gradients/batch_normalization_3/batchnorm/mul_2_grad/MulHtraining/Adam/gradients/batch_normalization_3/moments/Squeeze_grad/Shape*
T0*8
_class.
,*loc:@batch_normalization_3/moments/Squeeze*
Tshape0*
_output_shapes
:	�
�
training/Adam/gradients/AddN_8AddNLtraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/Reshape_1Htraining/Adam/gradients/batch_normalization_3/batchnorm/mul_2_grad/Mul_1*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/mul_1*
N*
_output_shapes	
:�
�
Dtraining/Adam/gradients/batch_normalization_3/batchnorm/mul_grad/MulMultraining/Adam/gradients/AddN_82batch_normalization_3/batchnorm/mul/ReadVariableOp*
_output_shapes	
:�*
T0*6
_class,
*(loc:@batch_normalization_3/batchnorm/mul
�
Ftraining/Adam/gradients/batch_normalization_3/batchnorm/mul_grad/Mul_1Multraining/Adam/gradients/AddN_8%batch_normalization_3/batchnorm/Rsqrt*
T0*6
_class,
*(loc:@batch_normalization_3/batchnorm/mul*
_output_shapes	
:�
�
 training/Adam/gradients/Switch_9Switchbatch_normalization_3/gamma"batch_normalization_3/cond/pred_id*
_output_shapes
: : *
T0*.
_class$
" loc:@batch_normalization_3/gamma
�
"training/Adam/gradients/Identity_9Identity"training/Adam/gradients/Switch_9:1*
T0*.
_class$
" loc:@batch_normalization_3/gamma*
_output_shapes
: 
�
'training/Adam/gradients/VariableShape_3VariableShape"training/Adam/gradients/Switch_9:1#^training/Adam/gradients/Identity_9*.
_class$
" loc:@batch_normalization_3/gamma*
out_type0*
_output_shapes
:
�
%training/Adam/gradients/zeros_9/ConstConst#^training/Adam/gradients/Identity_9*.
_class$
" loc:@batch_normalization_3/gamma*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/gradients/zeros_9Fill'training/Adam/gradients/VariableShape_3%training/Adam/gradients/zeros_9/Const*#
_output_shapes
:���������*
T0*.
_class$
" loc:@batch_normalization_3/gamma*

index_type0
�
etraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul/ReadVariableOp/Switch_grad/cond_gradMergeKtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_grad/Mul_1training/Adam/gradients/zeros_9*
T0*.
_class$
" loc:@batch_normalization_3/gamma*
N*%
_output_shapes
:���������: 
�
Ltraining/Adam/gradients/batch_normalization_3/batchnorm/Rsqrt_grad/RsqrtGrad	RsqrtGrad%batch_normalization_3/batchnorm/RsqrtDtraining/Adam/gradients/batch_normalization_3/batchnorm/mul_grad/Mul*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/Rsqrt*
_output_shapes	
:�
�
Vtraining/Adam/gradients/batch_normalization_3/batchnorm/add_grad/Sum/reduction_indicesConst*6
_class,
*(loc:@batch_normalization_3/batchnorm/add*
valueB: *
dtype0*
_output_shapes
:
�
Dtraining/Adam/gradients/batch_normalization_3/batchnorm/add_grad/SumSumLtraining/Adam/gradients/batch_normalization_3/batchnorm/Rsqrt_grad/RsqrtGradVtraining/Adam/gradients/batch_normalization_3/batchnorm/add_grad/Sum/reduction_indices*
	keep_dims( *

Tidx0*
T0*6
_class,
*(loc:@batch_normalization_3/batchnorm/add*
_output_shapes
: 
�
Ntraining/Adam/gradients/batch_normalization_3/batchnorm/add_grad/Reshape/shapeConst*6
_class,
*(loc:@batch_normalization_3/batchnorm/add*
valueB *
dtype0*
_output_shapes
: 
�
Htraining/Adam/gradients/batch_normalization_3/batchnorm/add_grad/ReshapeReshapeDtraining/Adam/gradients/batch_normalization_3/batchnorm/add_grad/SumNtraining/Adam/gradients/batch_normalization_3/batchnorm/add_grad/Reshape/shape*
T0*6
_class,
*(loc:@batch_normalization_3/batchnorm/add*
Tshape0*
_output_shapes
: 
�
training/Adam/gradients/AddN_9AddNetraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul/ReadVariableOp/Switch_grad/cond_gradFtraining/Adam/gradients/batch_normalization_3/batchnorm/mul_grad/Mul_1*
N*
_output_shapes	
:�*
T0*.
_class$
" loc:@batch_normalization_3/gamma
�
Jtraining/Adam/gradients/batch_normalization_3/moments/Squeeze_1_grad/ShapeConst*:
_class0
.,loc:@batch_normalization_3/moments/Squeeze_1*
valueB"   �   *
dtype0*
_output_shapes
:
�
Ltraining/Adam/gradients/batch_normalization_3/moments/Squeeze_1_grad/ReshapeReshapeLtraining/Adam/gradients/batch_normalization_3/batchnorm/Rsqrt_grad/RsqrtGradJtraining/Adam/gradients/batch_normalization_3/moments/Squeeze_1_grad/Shape*
T0*:
_class0
.,loc:@batch_normalization_3/moments/Squeeze_1*
Tshape0*
_output_shapes
:	�
�
Itraining/Adam/gradients/batch_normalization_3/moments/variance_grad/ShapeShape/batch_normalization_3/moments/SquaredDifference*
_output_shapes
:*
T0*9
_class/
-+loc:@batch_normalization_3/moments/variance*
out_type0
�
Htraining/Adam/gradients/batch_normalization_3/moments/variance_grad/SizeConst*9
_class/
-+loc:@batch_normalization_3/moments/variance*
value	B :*
dtype0*
_output_shapes
: 
�
Gtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/addAddV28batch_normalization_3/moments/variance/reduction_indicesHtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Size*
T0*9
_class/
-+loc:@batch_normalization_3/moments/variance*
_output_shapes
:
�
Gtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/modFloorModGtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/addHtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Size*
T0*9
_class/
-+loc:@batch_normalization_3/moments/variance*
_output_shapes
:
�
Ktraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Shape_1Const*9
_class/
-+loc:@batch_normalization_3/moments/variance*
valueB:*
dtype0*
_output_shapes
:
�
Otraining/Adam/gradients/batch_normalization_3/moments/variance_grad/range/startConst*
dtype0*
_output_shapes
: *9
_class/
-+loc:@batch_normalization_3/moments/variance*
value	B : 
�
Otraining/Adam/gradients/batch_normalization_3/moments/variance_grad/range/deltaConst*9
_class/
-+loc:@batch_normalization_3/moments/variance*
value	B :*
dtype0*
_output_shapes
: 
�
Itraining/Adam/gradients/batch_normalization_3/moments/variance_grad/rangeRangeOtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/range/startHtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/SizeOtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/range/delta*

Tidx0*9
_class/
-+loc:@batch_normalization_3/moments/variance*
_output_shapes
:
�
Ntraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Fill/valueConst*9
_class/
-+loc:@batch_normalization_3/moments/variance*
value	B :*
dtype0*
_output_shapes
: 
�
Htraining/Adam/gradients/batch_normalization_3/moments/variance_grad/FillFillKtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Shape_1Ntraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Fill/value*
T0*9
_class/
-+loc:@batch_normalization_3/moments/variance*

index_type0*
_output_shapes
:
�
Qtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/DynamicStitchDynamicStitchItraining/Adam/gradients/batch_normalization_3/moments/variance_grad/rangeGtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/modItraining/Adam/gradients/batch_normalization_3/moments/variance_grad/ShapeHtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Fill*
T0*9
_class/
-+loc:@batch_normalization_3/moments/variance*
N*
_output_shapes
:
�
Mtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Maximum/yConst*9
_class/
-+loc:@batch_normalization_3/moments/variance*
value	B :*
dtype0*
_output_shapes
: 
�
Ktraining/Adam/gradients/batch_normalization_3/moments/variance_grad/MaximumMaximumQtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/DynamicStitchMtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Maximum/y*
_output_shapes
:*
T0*9
_class/
-+loc:@batch_normalization_3/moments/variance
�
Ltraining/Adam/gradients/batch_normalization_3/moments/variance_grad/floordivFloorDivItraining/Adam/gradients/batch_normalization_3/moments/variance_grad/ShapeKtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Maximum*
T0*9
_class/
-+loc:@batch_normalization_3/moments/variance*
_output_shapes
:
�
Ktraining/Adam/gradients/batch_normalization_3/moments/variance_grad/ReshapeReshapeLtraining/Adam/gradients/batch_normalization_3/moments/Squeeze_1_grad/ReshapeQtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/DynamicStitch*0
_output_shapes
:������������������*
T0*9
_class/
-+loc:@batch_normalization_3/moments/variance*
Tshape0
�
Htraining/Adam/gradients/batch_normalization_3/moments/variance_grad/TileTileKtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/ReshapeLtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/floordiv*0
_output_shapes
:������������������*

Tmultiples0*
T0*9
_class/
-+loc:@batch_normalization_3/moments/variance
�
Ktraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Shape_2Shape/batch_normalization_3/moments/SquaredDifference*
T0*9
_class/
-+loc:@batch_normalization_3/moments/variance*
out_type0*
_output_shapes
:
�
Ktraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Shape_3Const*9
_class/
-+loc:@batch_normalization_3/moments/variance*
valueB"   �   *
dtype0*
_output_shapes
:
�
Itraining/Adam/gradients/batch_normalization_3/moments/variance_grad/ConstConst*9
_class/
-+loc:@batch_normalization_3/moments/variance*
valueB: *
dtype0*
_output_shapes
:
�
Htraining/Adam/gradients/batch_normalization_3/moments/variance_grad/ProdProdKtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Shape_2Itraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Const*
T0*9
_class/
-+loc:@batch_normalization_3/moments/variance*
_output_shapes
: *
	keep_dims( *

Tidx0
�
Ktraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Const_1Const*9
_class/
-+loc:@batch_normalization_3/moments/variance*
valueB: *
dtype0*
_output_shapes
:
�
Jtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Prod_1ProdKtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Shape_3Ktraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Const_1*
	keep_dims( *

Tidx0*
T0*9
_class/
-+loc:@batch_normalization_3/moments/variance*
_output_shapes
: 
�
Otraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Maximum_1/yConst*9
_class/
-+loc:@batch_normalization_3/moments/variance*
value	B :*
dtype0*
_output_shapes
: 
�
Mtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Maximum_1MaximumJtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Prod_1Otraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Maximum_1/y*
_output_shapes
: *
T0*9
_class/
-+loc:@batch_normalization_3/moments/variance
�
Ntraining/Adam/gradients/batch_normalization_3/moments/variance_grad/floordiv_1FloorDivHtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/ProdMtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Maximum_1*
T0*9
_class/
-+loc:@batch_normalization_3/moments/variance*
_output_shapes
: 
�
Htraining/Adam/gradients/batch_normalization_3/moments/variance_grad/CastCastNtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/floordiv_1*

SrcT0*9
_class/
-+loc:@batch_normalization_3/moments/variance*
Truncate( *

DstT0*
_output_shapes
: 
�
Ktraining/Adam/gradients/batch_normalization_3/moments/variance_grad/truedivRealDivHtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/TileHtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Cast*
T0*9
_class/
-+loc:@batch_normalization_3/moments/variance*(
_output_shapes
:����������
�
Straining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/scalarConstL^training/Adam/gradients/batch_normalization_3/moments/variance_grad/truediv*
dtype0*
_output_shapes
: *B
_class8
64loc:@batch_normalization_3/moments/SquaredDifference*
valueB
 *   @
�
Ptraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/MulMulStraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/scalarKtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/truediv*
T0*B
_class8
64loc:@batch_normalization_3/moments/SquaredDifference*(
_output_shapes
:����������
�
Ptraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/subSubdense_3/Relu*batch_normalization_3/moments/StopGradientL^training/Adam/gradients/batch_normalization_3/moments/variance_grad/truediv*
T0*B
_class8
64loc:@batch_normalization_3/moments/SquaredDifference*(
_output_shapes
:����������
�
Rtraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/mul_1MulPtraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/MulPtraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/sub*
T0*B
_class8
64loc:@batch_normalization_3/moments/SquaredDifference*(
_output_shapes
:����������
�
Rtraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/ShapeShapedense_3/Relu*
T0*B
_class8
64loc:@batch_normalization_3/moments/SquaredDifference*
out_type0*
_output_shapes
:
�
Ttraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/Shape_1Shape*batch_normalization_3/moments/StopGradient*
T0*B
_class8
64loc:@batch_normalization_3/moments/SquaredDifference*
out_type0*
_output_shapes
:
�
btraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgsRtraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/ShapeTtraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0*B
_class8
64loc:@batch_normalization_3/moments/SquaredDifference
�
Ptraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/SumSumRtraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/mul_1btraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/BroadcastGradientArgs*
T0*B
_class8
64loc:@batch_normalization_3/moments/SquaredDifference*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Ttraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/ReshapeReshapePtraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/SumRtraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/Shape*
T0*B
_class8
64loc:@batch_normalization_3/moments/SquaredDifference*
Tshape0*(
_output_shapes
:����������
�
Rtraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/Sum_1SumRtraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/mul_1dtraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*B
_class8
64loc:@batch_normalization_3/moments/SquaredDifference*
_output_shapes
:
�
Vtraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/Reshape_1ReshapeRtraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/Sum_1Ttraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/Shape_1*
T0*B
_class8
64loc:@batch_normalization_3/moments/SquaredDifference*
Tshape0*
_output_shapes
:	�
�
Ptraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/NegNegVtraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/Reshape_1*
T0*B
_class8
64loc:@batch_normalization_3/moments/SquaredDifference*
_output_shapes
:	�
�
Etraining/Adam/gradients/batch_normalization_3/moments/mean_grad/ShapeShapedense_3/Relu*
T0*5
_class+
)'loc:@batch_normalization_3/moments/mean*
out_type0*
_output_shapes
:
�
Dtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/SizeConst*5
_class+
)'loc:@batch_normalization_3/moments/mean*
value	B :*
dtype0*
_output_shapes
: 
�
Ctraining/Adam/gradients/batch_normalization_3/moments/mean_grad/addAddV24batch_normalization_3/moments/mean/reduction_indicesDtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/Size*
T0*5
_class+
)'loc:@batch_normalization_3/moments/mean*
_output_shapes
:
�
Ctraining/Adam/gradients/batch_normalization_3/moments/mean_grad/modFloorModCtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/addDtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/Size*
T0*5
_class+
)'loc:@batch_normalization_3/moments/mean*
_output_shapes
:
�
Gtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/Shape_1Const*5
_class+
)'loc:@batch_normalization_3/moments/mean*
valueB:*
dtype0*
_output_shapes
:
�
Ktraining/Adam/gradients/batch_normalization_3/moments/mean_grad/range/startConst*5
_class+
)'loc:@batch_normalization_3/moments/mean*
value	B : *
dtype0*
_output_shapes
: 
�
Ktraining/Adam/gradients/batch_normalization_3/moments/mean_grad/range/deltaConst*
dtype0*
_output_shapes
: *5
_class+
)'loc:@batch_normalization_3/moments/mean*
value	B :
�
Etraining/Adam/gradients/batch_normalization_3/moments/mean_grad/rangeRangeKtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/range/startDtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/SizeKtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/range/delta*

Tidx0*5
_class+
)'loc:@batch_normalization_3/moments/mean*
_output_shapes
:
�
Jtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/Fill/valueConst*5
_class+
)'loc:@batch_normalization_3/moments/mean*
value	B :*
dtype0*
_output_shapes
: 
�
Dtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/FillFillGtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/Shape_1Jtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/Fill/value*
T0*5
_class+
)'loc:@batch_normalization_3/moments/mean*

index_type0*
_output_shapes
:
�
Mtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/DynamicStitchDynamicStitchEtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/rangeCtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/modEtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/ShapeDtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/Fill*
T0*5
_class+
)'loc:@batch_normalization_3/moments/mean*
N*
_output_shapes
:
�
Itraining/Adam/gradients/batch_normalization_3/moments/mean_grad/Maximum/yConst*
dtype0*
_output_shapes
: *5
_class+
)'loc:@batch_normalization_3/moments/mean*
value	B :
�
Gtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/MaximumMaximumMtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/DynamicStitchItraining/Adam/gradients/batch_normalization_3/moments/mean_grad/Maximum/y*
T0*5
_class+
)'loc:@batch_normalization_3/moments/mean*
_output_shapes
:
�
Htraining/Adam/gradients/batch_normalization_3/moments/mean_grad/floordivFloorDivEtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/ShapeGtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/Maximum*
_output_shapes
:*
T0*5
_class+
)'loc:@batch_normalization_3/moments/mean
�
Gtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/ReshapeReshapeJtraining/Adam/gradients/batch_normalization_3/moments/Squeeze_grad/ReshapeMtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/DynamicStitch*
T0*5
_class+
)'loc:@batch_normalization_3/moments/mean*
Tshape0*0
_output_shapes
:������������������
�
Dtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/TileTileGtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/ReshapeHtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/floordiv*

Tmultiples0*
T0*5
_class+
)'loc:@batch_normalization_3/moments/mean*0
_output_shapes
:������������������
�
Gtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/Shape_2Shapedense_3/Relu*
T0*5
_class+
)'loc:@batch_normalization_3/moments/mean*
out_type0*
_output_shapes
:
�
Gtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/Shape_3Const*5
_class+
)'loc:@batch_normalization_3/moments/mean*
valueB"   �   *
dtype0*
_output_shapes
:
�
Etraining/Adam/gradients/batch_normalization_3/moments/mean_grad/ConstConst*5
_class+
)'loc:@batch_normalization_3/moments/mean*
valueB: *
dtype0*
_output_shapes
:
�
Dtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/ProdProdGtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/Shape_2Etraining/Adam/gradients/batch_normalization_3/moments/mean_grad/Const*
T0*5
_class+
)'loc:@batch_normalization_3/moments/mean*
_output_shapes
: *
	keep_dims( *

Tidx0
�
Gtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/Const_1Const*5
_class+
)'loc:@batch_normalization_3/moments/mean*
valueB: *
dtype0*
_output_shapes
:
�
Ftraining/Adam/gradients/batch_normalization_3/moments/mean_grad/Prod_1ProdGtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/Shape_3Gtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0*5
_class+
)'loc:@batch_normalization_3/moments/mean
�
Ktraining/Adam/gradients/batch_normalization_3/moments/mean_grad/Maximum_1/yConst*5
_class+
)'loc:@batch_normalization_3/moments/mean*
value	B :*
dtype0*
_output_shapes
: 
�
Itraining/Adam/gradients/batch_normalization_3/moments/mean_grad/Maximum_1MaximumFtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/Prod_1Ktraining/Adam/gradients/batch_normalization_3/moments/mean_grad/Maximum_1/y*
T0*5
_class+
)'loc:@batch_normalization_3/moments/mean*
_output_shapes
: 
�
Jtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/floordiv_1FloorDivDtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/ProdItraining/Adam/gradients/batch_normalization_3/moments/mean_grad/Maximum_1*
T0*5
_class+
)'loc:@batch_normalization_3/moments/mean*
_output_shapes
: 
�
Dtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/CastCastJtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/floordiv_1*

SrcT0*5
_class+
)'loc:@batch_normalization_3/moments/mean*
Truncate( *

DstT0*
_output_shapes
: 
�
Gtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/truedivRealDivDtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/TileDtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/Cast*
T0*5
_class+
)'loc:@batch_normalization_3/moments/mean*(
_output_shapes
:����������
�
training/Adam/gradients/AddN_10AddNXtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_1/Switch_grad/cond_gradJtraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/ReshapeTtraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/ReshapeGtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/truediv*
T0*
_class
loc:@dense_3/Relu*
N*(
_output_shapes
:����������
�
2training/Adam/gradients/dense_3/Relu_grad/ReluGradReluGradtraining/Adam/gradients/AddN_10dense_3/Relu*
T0*
_class
loc:@dense_3/Relu*(
_output_shapes
:����������
�
8training/Adam/gradients/dense_3/BiasAdd_grad/BiasAddGradBiasAddGrad2training/Adam/gradients/dense_3/Relu_grad/ReluGrad*
T0*"
_class
loc:@dense_3/BiasAdd*
data_formatNHWC*
_output_shapes	
:�
�
2training/Adam/gradients/dense_3/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_3/Relu_grad/ReluGraddense_3/MatMul/ReadVariableOp*
transpose_b(*
T0*!
_class
loc:@dense_3/MatMul*
transpose_a( *(
_output_shapes
:����������
�
4training/Adam/gradients/dense_3/MatMul_grad/MatMul_1MatMuldropout_1/cond/Merge2training/Adam/gradients/dense_3/Relu_grad/ReluGrad*
transpose_b( *
T0*!
_class
loc:@dense_3/MatMul*
transpose_a(* 
_output_shapes
:
��
�
;training/Adam/gradients/dropout_1/cond/Merge_grad/cond_gradSwitch2training/Adam/gradients/dense_3/MatMul_grad/MatMuldropout_1/cond/pred_id*<
_output_shapes*
(:����������:����������*
T0*!
_class
loc:@dense_3/MatMul
�
!training/Adam/gradients/Switch_10Switch batch_normalization_2/cond/Mergedropout_1/cond/pred_id*
T0*3
_class)
'%loc:@batch_normalization_2/cond/Merge*<
_output_shapes*
(:����������:����������
�
#training/Adam/gradients/Identity_10Identity#training/Adam/gradients/Switch_10:1*(
_output_shapes
:����������*
T0*3
_class)
'%loc:@batch_normalization_2/cond/Merge
�
training/Adam/gradients/Shape_7Shape#training/Adam/gradients/Switch_10:1*
T0*3
_class)
'%loc:@batch_normalization_2/cond/Merge*
out_type0*
_output_shapes
:
�
&training/Adam/gradients/zeros_10/ConstConst$^training/Adam/gradients/Identity_10*3
_class)
'%loc:@batch_normalization_2/cond/Merge*
valueB
 *    *
dtype0*
_output_shapes
: 
�
 training/Adam/gradients/zeros_10Filltraining/Adam/gradients/Shape_7&training/Adam/gradients/zeros_10/Const*
T0*3
_class)
'%loc:@batch_normalization_2/cond/Merge*

index_type0*(
_output_shapes
:����������
�
>training/Adam/gradients/dropout_1/cond/Switch_1_grad/cond_gradMerge;training/Adam/gradients/dropout_1/cond/Merge_grad/cond_grad training/Adam/gradients/zeros_10*
T0*3
_class)
'%loc:@batch_normalization_2/cond/Merge*
N**
_output_shapes
:����������: 
�
?training/Adam/gradients/dropout_1/cond/dropout/mul_1_grad/ShapeShapedropout_1/cond/dropout/mul*
_output_shapes
:*
T0*/
_class%
#!loc:@dropout_1/cond/dropout/mul_1*
out_type0
�
Atraining/Adam/gradients/dropout_1/cond/dropout/mul_1_grad/Shape_1Shapedropout_1/cond/dropout/Cast*
T0*/
_class%
#!loc:@dropout_1/cond/dropout/mul_1*
out_type0*
_output_shapes
:
�
Otraining/Adam/gradients/dropout_1/cond/dropout/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs?training/Adam/gradients/dropout_1/cond/dropout/mul_1_grad/ShapeAtraining/Adam/gradients/dropout_1/cond/dropout/mul_1_grad/Shape_1*
T0*/
_class%
#!loc:@dropout_1/cond/dropout/mul_1*2
_output_shapes 
:���������:���������
�
=training/Adam/gradients/dropout_1/cond/dropout/mul_1_grad/MulMul=training/Adam/gradients/dropout_1/cond/Merge_grad/cond_grad:1dropout_1/cond/dropout/Cast*(
_output_shapes
:����������*
T0*/
_class%
#!loc:@dropout_1/cond/dropout/mul_1
�
=training/Adam/gradients/dropout_1/cond/dropout/mul_1_grad/SumSum=training/Adam/gradients/dropout_1/cond/dropout/mul_1_grad/MulOtraining/Adam/gradients/dropout_1/cond/dropout/mul_1_grad/BroadcastGradientArgs*
T0*/
_class%
#!loc:@dropout_1/cond/dropout/mul_1*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Atraining/Adam/gradients/dropout_1/cond/dropout/mul_1_grad/ReshapeReshape=training/Adam/gradients/dropout_1/cond/dropout/mul_1_grad/Sum?training/Adam/gradients/dropout_1/cond/dropout/mul_1_grad/Shape*
T0*/
_class%
#!loc:@dropout_1/cond/dropout/mul_1*
Tshape0*(
_output_shapes
:����������
�
?training/Adam/gradients/dropout_1/cond/dropout/mul_1_grad/Mul_1Muldropout_1/cond/dropout/mul=training/Adam/gradients/dropout_1/cond/Merge_grad/cond_grad:1*(
_output_shapes
:����������*
T0*/
_class%
#!loc:@dropout_1/cond/dropout/mul_1
�
?training/Adam/gradients/dropout_1/cond/dropout/mul_1_grad/Sum_1Sum?training/Adam/gradients/dropout_1/cond/dropout/mul_1_grad/Mul_1Qtraining/Adam/gradients/dropout_1/cond/dropout/mul_1_grad/BroadcastGradientArgs:1*
T0*/
_class%
#!loc:@dropout_1/cond/dropout/mul_1*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Ctraining/Adam/gradients/dropout_1/cond/dropout/mul_1_grad/Reshape_1Reshape?training/Adam/gradients/dropout_1/cond/dropout/mul_1_grad/Sum_1Atraining/Adam/gradients/dropout_1/cond/dropout/mul_1_grad/Shape_1*
T0*/
_class%
#!loc:@dropout_1/cond/dropout/mul_1*
Tshape0*(
_output_shapes
:����������
�
=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/ShapeShape%dropout_1/cond/dropout/Shape/Switch:1*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*
out_type0*
_output_shapes
:
�
?training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Shape_1Shapedropout_1/cond/dropout/truediv*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*
out_type0*
_output_shapes
: 
�
Mtraining/Adam/gradients/dropout_1/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Shape?training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul
�
;training/Adam/gradients/dropout_1/cond/dropout/mul_grad/MulMulAtraining/Adam/gradients/dropout_1/cond/dropout/mul_1_grad/Reshapedropout_1/cond/dropout/truediv*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*(
_output_shapes
:����������
�
;training/Adam/gradients/dropout_1/cond/dropout/mul_grad/SumSum;training/Adam/gradients/dropout_1/cond/dropout/mul_grad/MulMtraining/Adam/gradients/dropout_1/cond/dropout/mul_grad/BroadcastGradientArgs*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*
_output_shapes
:*
	keep_dims( *

Tidx0
�
?training/Adam/gradients/dropout_1/cond/dropout/mul_grad/ReshapeReshape;training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Sum=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Shape*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*
Tshape0*(
_output_shapes
:����������
�
=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Mul_1Mul%dropout_1/cond/dropout/Shape/Switch:1Atraining/Adam/gradients/dropout_1/cond/dropout/mul_1_grad/Reshape*(
_output_shapes
:����������*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul
�
=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Sum_1Sum=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Mul_1Otraining/Adam/gradients/dropout_1/cond/dropout/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul
�
Atraining/Adam/gradients/dropout_1/cond/dropout/mul_grad/Reshape_1Reshape=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Sum_1?training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Shape_1*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*
Tshape0*
_output_shapes
: 
�
!training/Adam/gradients/Switch_11Switch batch_normalization_2/cond/Mergedropout_1/cond/pred_id*
T0*3
_class)
'%loc:@batch_normalization_2/cond/Merge*<
_output_shapes*
(:����������:����������
�
#training/Adam/gradients/Identity_11Identity!training/Adam/gradients/Switch_11*(
_output_shapes
:����������*
T0*3
_class)
'%loc:@batch_normalization_2/cond/Merge
�
training/Adam/gradients/Shape_8Shape!training/Adam/gradients/Switch_11*
T0*3
_class)
'%loc:@batch_normalization_2/cond/Merge*
out_type0*
_output_shapes
:
�
&training/Adam/gradients/zeros_11/ConstConst$^training/Adam/gradients/Identity_11*3
_class)
'%loc:@batch_normalization_2/cond/Merge*
valueB
 *    *
dtype0*
_output_shapes
: 
�
 training/Adam/gradients/zeros_11Filltraining/Adam/gradients/Shape_8&training/Adam/gradients/zeros_11/Const*
T0*3
_class)
'%loc:@batch_normalization_2/cond/Merge*

index_type0*(
_output_shapes
:����������
�
Jtraining/Adam/gradients/dropout_1/cond/dropout/Shape/Switch_grad/cond_gradMerge training/Adam/gradients/zeros_11?training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Reshape*
T0*3
_class)
'%loc:@batch_normalization_2/cond/Merge*
N**
_output_shapes
:����������: 
�
training/Adam/gradients/AddN_11AddN>training/Adam/gradients/dropout_1/cond/Switch_1_grad/cond_gradJtraining/Adam/gradients/dropout_1/cond/dropout/Shape/Switch_grad/cond_grad*
T0*3
_class)
'%loc:@batch_normalization_2/cond/Merge*
N*(
_output_shapes
:����������
�
Gtraining/Adam/gradients/batch_normalization_2/cond/Merge_grad/cond_gradSwitchtraining/Adam/gradients/AddN_11"batch_normalization_2/cond/pred_id*
T0*3
_class)
'%loc:@batch_normalization_2/cond/Merge*<
_output_shapes*
(:����������:����������
�
Mtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/ShapeShape*batch_normalization_2/cond/batchnorm/mul_1*
_output_shapes
:*
T0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/add_1*
out_type0
�
Otraining/Adam/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/Shape_1Shape(batch_normalization_2/cond/batchnorm/sub*
T0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/add_1*
out_type0*
_output_shapes
:
�
]training/Adam/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsMtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/ShapeOtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/Shape_1*
T0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/add_1*2
_output_shapes 
:���������:���������
�
Ktraining/Adam/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/SumSumGtraining/Adam/gradients/batch_normalization_2/cond/Merge_grad/cond_grad]training/Adam/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/BroadcastGradientArgs*
T0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/add_1*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Otraining/Adam/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/ReshapeReshapeKtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/SumMtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/Shape*
T0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/add_1*
Tshape0*(
_output_shapes
:����������
�
Mtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/Sum_1SumGtraining/Adam/gradients/batch_normalization_2/cond/Merge_grad/cond_grad_training/Adam/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/add_1*
_output_shapes
:
�
Qtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/Reshape_1ReshapeMtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/Sum_1Otraining/Adam/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/Shape_1*
_output_shapes	
:�*
T0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/add_1*
Tshape0
�
!training/Adam/gradients/Switch_12Switch%batch_normalization_2/batchnorm/add_1"batch_normalization_2/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/add_1*<
_output_shapes*
(:����������:����������
�
#training/Adam/gradients/Identity_12Identity!training/Adam/gradients/Switch_12*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/add_1*(
_output_shapes
:����������
�
training/Adam/gradients/Shape_9Shape!training/Adam/gradients/Switch_12*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/add_1*
out_type0*
_output_shapes
:
�
&training/Adam/gradients/zeros_12/ConstConst$^training/Adam/gradients/Identity_12*8
_class.
,*loc:@batch_normalization_2/batchnorm/add_1*
valueB
 *    *
dtype0*
_output_shapes
: 
�
 training/Adam/gradients/zeros_12Filltraining/Adam/gradients/Shape_9&training/Adam/gradients/zeros_12/Const*(
_output_shapes
:����������*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/add_1*

index_type0
�
Jtraining/Adam/gradients/batch_normalization_2/cond/Switch_1_grad/cond_gradMerge training/Adam/gradients/zeros_12Itraining/Adam/gradients/batch_normalization_2/cond/Merge_grad/cond_grad:1*
N**
_output_shapes
:����������: *
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/add_1
�
Mtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/ShapeShape1batch_normalization_2/cond/batchnorm/mul_1/Switch*
T0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/mul_1*
out_type0*
_output_shapes
:
�
Otraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/Shape_1Shape(batch_normalization_2/cond/batchnorm/mul*
T0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/mul_1*
out_type0*
_output_shapes
:
�
]training/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsMtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/ShapeOtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/Shape_1*
T0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/mul_1*2
_output_shapes 
:���������:���������
�
Ktraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/MulMulOtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/Reshape(batch_normalization_2/cond/batchnorm/mul*
T0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/mul_1*(
_output_shapes
:����������
�
Ktraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/SumSumKtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/Mul]training/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/BroadcastGradientArgs*
T0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/mul_1*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Otraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/ReshapeReshapeKtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/SumMtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/Shape*(
_output_shapes
:����������*
T0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/mul_1*
Tshape0
�
Mtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/Mul_1Mul1batch_normalization_2/cond/batchnorm/mul_1/SwitchOtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/Reshape*
T0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/mul_1*(
_output_shapes
:����������
�
Mtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/Sum_1SumMtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/Mul_1_training/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/mul_1*
_output_shapes
:
�
Qtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/Reshape_1ReshapeMtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/Sum_1Otraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/Shape_1*
T0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/mul_1*
Tshape0*
_output_shapes	
:�
�
Itraining/Adam/gradients/batch_normalization_2/cond/batchnorm/sub_grad/NegNegQtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/Reshape_1*
T0*;
_class1
/-loc:@batch_normalization_2/cond/batchnorm/sub*
_output_shapes	
:�
�
Htraining/Adam/gradients/batch_normalization_2/batchnorm/add_1_grad/ShapeShape%batch_normalization_2/batchnorm/mul_1*
_output_shapes
:*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/add_1*
out_type0
�
Jtraining/Adam/gradients/batch_normalization_2/batchnorm/add_1_grad/Shape_1Shape#batch_normalization_2/batchnorm/sub*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/add_1*
out_type0*
_output_shapes
:
�
Xtraining/Adam/gradients/batch_normalization_2/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsHtraining/Adam/gradients/batch_normalization_2/batchnorm/add_1_grad/ShapeJtraining/Adam/gradients/batch_normalization_2/batchnorm/add_1_grad/Shape_1*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/add_1*2
_output_shapes 
:���������:���������
�
Ftraining/Adam/gradients/batch_normalization_2/batchnorm/add_1_grad/SumSumJtraining/Adam/gradients/batch_normalization_2/cond/Switch_1_grad/cond_gradXtraining/Adam/gradients/batch_normalization_2/batchnorm/add_1_grad/BroadcastGradientArgs*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/add_1*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Jtraining/Adam/gradients/batch_normalization_2/batchnorm/add_1_grad/ReshapeReshapeFtraining/Adam/gradients/batch_normalization_2/batchnorm/add_1_grad/SumHtraining/Adam/gradients/batch_normalization_2/batchnorm/add_1_grad/Shape*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/add_1*
Tshape0*(
_output_shapes
:����������
�
Htraining/Adam/gradients/batch_normalization_2/batchnorm/add_1_grad/Sum_1SumJtraining/Adam/gradients/batch_normalization_2/cond/Switch_1_grad/cond_gradZtraining/Adam/gradients/batch_normalization_2/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/add_1*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Ltraining/Adam/gradients/batch_normalization_2/batchnorm/add_1_grad/Reshape_1ReshapeHtraining/Adam/gradients/batch_normalization_2/batchnorm/add_1_grad/Sum_1Jtraining/Adam/gradients/batch_normalization_2/batchnorm/add_1_grad/Shape_1*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/add_1*
Tshape0*
_output_shapes	
:�
�
!training/Adam/gradients/Switch_13Switchdense_2/Relu"batch_normalization_2/cond/pred_id*
T0*
_class
loc:@dense_2/Relu*<
_output_shapes*
(:����������:����������
�
#training/Adam/gradients/Identity_13Identity#training/Adam/gradients/Switch_13:1*(
_output_shapes
:����������*
T0*
_class
loc:@dense_2/Relu
�
 training/Adam/gradients/Shape_10Shape#training/Adam/gradients/Switch_13:1*
T0*
_class
loc:@dense_2/Relu*
out_type0*
_output_shapes
:
�
&training/Adam/gradients/zeros_13/ConstConst$^training/Adam/gradients/Identity_13*
_class
loc:@dense_2/Relu*
valueB
 *    *
dtype0*
_output_shapes
: 
�
 training/Adam/gradients/zeros_13Fill training/Adam/gradients/Shape_10&training/Adam/gradients/zeros_13/Const*
T0*
_class
loc:@dense_2/Relu*

index_type0*(
_output_shapes
:����������
�
Xtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1/Switch_grad/cond_gradMergeOtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/Reshape training/Adam/gradients/zeros_13*
N**
_output_shapes
:����������: *
T0*
_class
loc:@dense_2/Relu
�
Ktraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_2_grad/MulMulItraining/Adam/gradients/batch_normalization_2/cond/batchnorm/sub_grad/Neg(batch_normalization_2/cond/batchnorm/mul*
T0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/mul_2*
_output_shapes	
:�
�
Mtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_2_grad/Mul_1MulItraining/Adam/gradients/batch_normalization_2/cond/batchnorm/sub_grad/Neg5batch_normalization_2/cond/batchnorm/ReadVariableOp_1*
T0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/mul_2*
_output_shapes	
:�
�
Htraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/ShapeShapedense_2/Relu*
_output_shapes
:*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/mul_1*
out_type0
�
Jtraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/Shape_1Shape#batch_normalization_2/batchnorm/mul*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/mul_1*
out_type0*
_output_shapes
:
�
Xtraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsHtraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/ShapeJtraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/mul_1
�
Ftraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/MulMulJtraining/Adam/gradients/batch_normalization_2/batchnorm/add_1_grad/Reshape#batch_normalization_2/batchnorm/mul*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/mul_1*(
_output_shapes
:����������
�
Ftraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/SumSumFtraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/MulXtraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/BroadcastGradientArgs*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/mul_1*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Jtraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/ReshapeReshapeFtraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/SumHtraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/Shape*(
_output_shapes
:����������*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/mul_1*
Tshape0
�
Htraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/Mul_1Muldense_2/ReluJtraining/Adam/gradients/batch_normalization_2/batchnorm/add_1_grad/Reshape*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/mul_1*(
_output_shapes
:����������
�
Htraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/Sum_1SumHtraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/Mul_1Ztraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/mul_1*
_output_shapes
:
�
Ltraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/Reshape_1ReshapeHtraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/Sum_1Jtraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/Shape_1*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/mul_1*
Tshape0*
_output_shapes	
:�
�
Dtraining/Adam/gradients/batch_normalization_2/batchnorm/sub_grad/NegNegLtraining/Adam/gradients/batch_normalization_2/batchnorm/add_1_grad/Reshape_1*
T0*6
_class,
*(loc:@batch_normalization_2/batchnorm/sub*
_output_shapes	
:�
�
!training/Adam/gradients/Switch_14Switchbatch_normalization_2/beta"batch_normalization_2/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_2/beta*
_output_shapes
: : 
�
#training/Adam/gradients/Identity_14Identity#training/Adam/gradients/Switch_14:1*
T0*-
_class#
!loc:@batch_normalization_2/beta*
_output_shapes
: 
�
'training/Adam/gradients/VariableShape_4VariableShape#training/Adam/gradients/Switch_14:1$^training/Adam/gradients/Identity_14*
_output_shapes
:*-
_class#
!loc:@batch_normalization_2/beta*
out_type0
�
&training/Adam/gradients/zeros_14/ConstConst$^training/Adam/gradients/Identity_14*
dtype0*
_output_shapes
: *-
_class#
!loc:@batch_normalization_2/beta*
valueB
 *    
�
 training/Adam/gradients/zeros_14Fill'training/Adam/gradients/VariableShape_4&training/Adam/gradients/zeros_14/Const*
T0*-
_class#
!loc:@batch_normalization_2/beta*

index_type0*#
_output_shapes
:���������
�
ctraining/Adam/gradients/batch_normalization_2/cond/batchnorm/ReadVariableOp_2/Switch_grad/cond_gradMergeQtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/Reshape_1 training/Adam/gradients/zeros_14*
T0*-
_class#
!loc:@batch_normalization_2/beta*
N*%
_output_shapes
:���������: 
�
training/Adam/gradients/AddN_12AddNQtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/Reshape_1Mtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_2_grad/Mul_1*
T0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/mul_1*
N*
_output_shapes	
:�
�
Itraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_grad/MulMultraining/Adam/gradients/AddN_127batch_normalization_2/cond/batchnorm/mul/ReadVariableOp*
T0*;
_class1
/-loc:@batch_normalization_2/cond/batchnorm/mul*
_output_shapes	
:�
�
Ktraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_grad/Mul_1Multraining/Adam/gradients/AddN_12*batch_normalization_2/cond/batchnorm/Rsqrt*
T0*;
_class1
/-loc:@batch_normalization_2/cond/batchnorm/mul*
_output_shapes	
:�
�
Ftraining/Adam/gradients/batch_normalization_2/batchnorm/mul_2_grad/MulMulDtraining/Adam/gradients/batch_normalization_2/batchnorm/sub_grad/Neg#batch_normalization_2/batchnorm/mul*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/mul_2*
_output_shapes	
:�
�
Htraining/Adam/gradients/batch_normalization_2/batchnorm/mul_2_grad/Mul_1MulDtraining/Adam/gradients/batch_normalization_2/batchnorm/sub_grad/Neg%batch_normalization_2/moments/Squeeze*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/mul_2*
_output_shapes	
:�
�
training/Adam/gradients/AddN_13AddNctraining/Adam/gradients/batch_normalization_2/cond/batchnorm/ReadVariableOp_2/Switch_grad/cond_gradLtraining/Adam/gradients/batch_normalization_2/batchnorm/add_1_grad/Reshape_1*
T0*-
_class#
!loc:@batch_normalization_2/beta*
N*
_output_shapes	
:�
�
Htraining/Adam/gradients/batch_normalization_2/moments/Squeeze_grad/ShapeConst*8
_class.
,*loc:@batch_normalization_2/moments/Squeeze*
valueB"   �   *
dtype0*
_output_shapes
:
�
Jtraining/Adam/gradients/batch_normalization_2/moments/Squeeze_grad/ReshapeReshapeFtraining/Adam/gradients/batch_normalization_2/batchnorm/mul_2_grad/MulHtraining/Adam/gradients/batch_normalization_2/moments/Squeeze_grad/Shape*
_output_shapes
:	�*
T0*8
_class.
,*loc:@batch_normalization_2/moments/Squeeze*
Tshape0
�
training/Adam/gradients/AddN_14AddNLtraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/Reshape_1Htraining/Adam/gradients/batch_normalization_2/batchnorm/mul_2_grad/Mul_1*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/mul_1*
N*
_output_shapes	
:�
�
Dtraining/Adam/gradients/batch_normalization_2/batchnorm/mul_grad/MulMultraining/Adam/gradients/AddN_142batch_normalization_2/batchnorm/mul/ReadVariableOp*
_output_shapes	
:�*
T0*6
_class,
*(loc:@batch_normalization_2/batchnorm/mul
�
Ftraining/Adam/gradients/batch_normalization_2/batchnorm/mul_grad/Mul_1Multraining/Adam/gradients/AddN_14%batch_normalization_2/batchnorm/Rsqrt*
T0*6
_class,
*(loc:@batch_normalization_2/batchnorm/mul*
_output_shapes	
:�
�
!training/Adam/gradients/Switch_15Switchbatch_normalization_2/gamma"batch_normalization_2/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_2/gamma*
_output_shapes
: : 
�
#training/Adam/gradients/Identity_15Identity#training/Adam/gradients/Switch_15:1*
T0*.
_class$
" loc:@batch_normalization_2/gamma*
_output_shapes
: 
�
'training/Adam/gradients/VariableShape_5VariableShape#training/Adam/gradients/Switch_15:1$^training/Adam/gradients/Identity_15*.
_class$
" loc:@batch_normalization_2/gamma*
out_type0*
_output_shapes
:
�
&training/Adam/gradients/zeros_15/ConstConst$^training/Adam/gradients/Identity_15*.
_class$
" loc:@batch_normalization_2/gamma*
valueB
 *    *
dtype0*
_output_shapes
: 
�
 training/Adam/gradients/zeros_15Fill'training/Adam/gradients/VariableShape_5&training/Adam/gradients/zeros_15/Const*
T0*.
_class$
" loc:@batch_normalization_2/gamma*

index_type0*#
_output_shapes
:���������
�
etraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul/ReadVariableOp/Switch_grad/cond_gradMergeKtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_grad/Mul_1 training/Adam/gradients/zeros_15*
T0*.
_class$
" loc:@batch_normalization_2/gamma*
N*%
_output_shapes
:���������: 
�
Ltraining/Adam/gradients/batch_normalization_2/batchnorm/Rsqrt_grad/RsqrtGrad	RsqrtGrad%batch_normalization_2/batchnorm/RsqrtDtraining/Adam/gradients/batch_normalization_2/batchnorm/mul_grad/Mul*
_output_shapes	
:�*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/Rsqrt
�
Vtraining/Adam/gradients/batch_normalization_2/batchnorm/add_grad/Sum/reduction_indicesConst*
dtype0*
_output_shapes
:*6
_class,
*(loc:@batch_normalization_2/batchnorm/add*
valueB: 
�
Dtraining/Adam/gradients/batch_normalization_2/batchnorm/add_grad/SumSumLtraining/Adam/gradients/batch_normalization_2/batchnorm/Rsqrt_grad/RsqrtGradVtraining/Adam/gradients/batch_normalization_2/batchnorm/add_grad/Sum/reduction_indices*
	keep_dims( *

Tidx0*
T0*6
_class,
*(loc:@batch_normalization_2/batchnorm/add*
_output_shapes
: 
�
Ntraining/Adam/gradients/batch_normalization_2/batchnorm/add_grad/Reshape/shapeConst*6
_class,
*(loc:@batch_normalization_2/batchnorm/add*
valueB *
dtype0*
_output_shapes
: 
�
Htraining/Adam/gradients/batch_normalization_2/batchnorm/add_grad/ReshapeReshapeDtraining/Adam/gradients/batch_normalization_2/batchnorm/add_grad/SumNtraining/Adam/gradients/batch_normalization_2/batchnorm/add_grad/Reshape/shape*
_output_shapes
: *
T0*6
_class,
*(loc:@batch_normalization_2/batchnorm/add*
Tshape0
�
training/Adam/gradients/AddN_15AddNetraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul/ReadVariableOp/Switch_grad/cond_gradFtraining/Adam/gradients/batch_normalization_2/batchnorm/mul_grad/Mul_1*
T0*.
_class$
" loc:@batch_normalization_2/gamma*
N*
_output_shapes	
:�
�
Jtraining/Adam/gradients/batch_normalization_2/moments/Squeeze_1_grad/ShapeConst*:
_class0
.,loc:@batch_normalization_2/moments/Squeeze_1*
valueB"   �   *
dtype0*
_output_shapes
:
�
Ltraining/Adam/gradients/batch_normalization_2/moments/Squeeze_1_grad/ReshapeReshapeLtraining/Adam/gradients/batch_normalization_2/batchnorm/Rsqrt_grad/RsqrtGradJtraining/Adam/gradients/batch_normalization_2/moments/Squeeze_1_grad/Shape*
_output_shapes
:	�*
T0*:
_class0
.,loc:@batch_normalization_2/moments/Squeeze_1*
Tshape0
�
Itraining/Adam/gradients/batch_normalization_2/moments/variance_grad/ShapeShape/batch_normalization_2/moments/SquaredDifference*
_output_shapes
:*
T0*9
_class/
-+loc:@batch_normalization_2/moments/variance*
out_type0
�
Htraining/Adam/gradients/batch_normalization_2/moments/variance_grad/SizeConst*9
_class/
-+loc:@batch_normalization_2/moments/variance*
value	B :*
dtype0*
_output_shapes
: 
�
Gtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/addAddV28batch_normalization_2/moments/variance/reduction_indicesHtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/Size*
T0*9
_class/
-+loc:@batch_normalization_2/moments/variance*
_output_shapes
:
�
Gtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/modFloorModGtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/addHtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/Size*
T0*9
_class/
-+loc:@batch_normalization_2/moments/variance*
_output_shapes
:
�
Ktraining/Adam/gradients/batch_normalization_2/moments/variance_grad/Shape_1Const*
dtype0*
_output_shapes
:*9
_class/
-+loc:@batch_normalization_2/moments/variance*
valueB:
�
Otraining/Adam/gradients/batch_normalization_2/moments/variance_grad/range/startConst*9
_class/
-+loc:@batch_normalization_2/moments/variance*
value	B : *
dtype0*
_output_shapes
: 
�
Otraining/Adam/gradients/batch_normalization_2/moments/variance_grad/range/deltaConst*9
_class/
-+loc:@batch_normalization_2/moments/variance*
value	B :*
dtype0*
_output_shapes
: 
�
Itraining/Adam/gradients/batch_normalization_2/moments/variance_grad/rangeRangeOtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/range/startHtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/SizeOtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/range/delta*

Tidx0*9
_class/
-+loc:@batch_normalization_2/moments/variance*
_output_shapes
:
�
Ntraining/Adam/gradients/batch_normalization_2/moments/variance_grad/Fill/valueConst*9
_class/
-+loc:@batch_normalization_2/moments/variance*
value	B :*
dtype0*
_output_shapes
: 
�
Htraining/Adam/gradients/batch_normalization_2/moments/variance_grad/FillFillKtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/Shape_1Ntraining/Adam/gradients/batch_normalization_2/moments/variance_grad/Fill/value*
_output_shapes
:*
T0*9
_class/
-+loc:@batch_normalization_2/moments/variance*

index_type0
�
Qtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/DynamicStitchDynamicStitchItraining/Adam/gradients/batch_normalization_2/moments/variance_grad/rangeGtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/modItraining/Adam/gradients/batch_normalization_2/moments/variance_grad/ShapeHtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/Fill*
N*
_output_shapes
:*
T0*9
_class/
-+loc:@batch_normalization_2/moments/variance
�
Mtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/Maximum/yConst*9
_class/
-+loc:@batch_normalization_2/moments/variance*
value	B :*
dtype0*
_output_shapes
: 
�
Ktraining/Adam/gradients/batch_normalization_2/moments/variance_grad/MaximumMaximumQtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/DynamicStitchMtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/Maximum/y*
_output_shapes
:*
T0*9
_class/
-+loc:@batch_normalization_2/moments/variance
�
Ltraining/Adam/gradients/batch_normalization_2/moments/variance_grad/floordivFloorDivItraining/Adam/gradients/batch_normalization_2/moments/variance_grad/ShapeKtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/Maximum*
_output_shapes
:*
T0*9
_class/
-+loc:@batch_normalization_2/moments/variance
�
Ktraining/Adam/gradients/batch_normalization_2/moments/variance_grad/ReshapeReshapeLtraining/Adam/gradients/batch_normalization_2/moments/Squeeze_1_grad/ReshapeQtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/DynamicStitch*
T0*9
_class/
-+loc:@batch_normalization_2/moments/variance*
Tshape0*0
_output_shapes
:������������������
�
Htraining/Adam/gradients/batch_normalization_2/moments/variance_grad/TileTileKtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/ReshapeLtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/floordiv*
T0*9
_class/
-+loc:@batch_normalization_2/moments/variance*0
_output_shapes
:������������������*

Tmultiples0
�
Ktraining/Adam/gradients/batch_normalization_2/moments/variance_grad/Shape_2Shape/batch_normalization_2/moments/SquaredDifference*
_output_shapes
:*
T0*9
_class/
-+loc:@batch_normalization_2/moments/variance*
out_type0
�
Ktraining/Adam/gradients/batch_normalization_2/moments/variance_grad/Shape_3Const*9
_class/
-+loc:@batch_normalization_2/moments/variance*
valueB"   �   *
dtype0*
_output_shapes
:
�
Itraining/Adam/gradients/batch_normalization_2/moments/variance_grad/ConstConst*9
_class/
-+loc:@batch_normalization_2/moments/variance*
valueB: *
dtype0*
_output_shapes
:
�
Htraining/Adam/gradients/batch_normalization_2/moments/variance_grad/ProdProdKtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/Shape_2Itraining/Adam/gradients/batch_normalization_2/moments/variance_grad/Const*
T0*9
_class/
-+loc:@batch_normalization_2/moments/variance*
_output_shapes
: *
	keep_dims( *

Tidx0
�
Ktraining/Adam/gradients/batch_normalization_2/moments/variance_grad/Const_1Const*
dtype0*
_output_shapes
:*9
_class/
-+loc:@batch_normalization_2/moments/variance*
valueB: 
�
Jtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/Prod_1ProdKtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/Shape_3Ktraining/Adam/gradients/batch_normalization_2/moments/variance_grad/Const_1*
T0*9
_class/
-+loc:@batch_normalization_2/moments/variance*
_output_shapes
: *
	keep_dims( *

Tidx0
�
Otraining/Adam/gradients/batch_normalization_2/moments/variance_grad/Maximum_1/yConst*9
_class/
-+loc:@batch_normalization_2/moments/variance*
value	B :*
dtype0*
_output_shapes
: 
�
Mtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/Maximum_1MaximumJtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/Prod_1Otraining/Adam/gradients/batch_normalization_2/moments/variance_grad/Maximum_1/y*
T0*9
_class/
-+loc:@batch_normalization_2/moments/variance*
_output_shapes
: 
�
Ntraining/Adam/gradients/batch_normalization_2/moments/variance_grad/floordiv_1FloorDivHtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/ProdMtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/Maximum_1*
T0*9
_class/
-+loc:@batch_normalization_2/moments/variance*
_output_shapes
: 
�
Htraining/Adam/gradients/batch_normalization_2/moments/variance_grad/CastCastNtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/floordiv_1*

SrcT0*9
_class/
-+loc:@batch_normalization_2/moments/variance*
Truncate( *

DstT0*
_output_shapes
: 
�
Ktraining/Adam/gradients/batch_normalization_2/moments/variance_grad/truedivRealDivHtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/TileHtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/Cast*(
_output_shapes
:����������*
T0*9
_class/
-+loc:@batch_normalization_2/moments/variance
�
Straining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/scalarConstL^training/Adam/gradients/batch_normalization_2/moments/variance_grad/truediv*B
_class8
64loc:@batch_normalization_2/moments/SquaredDifference*
valueB
 *   @*
dtype0*
_output_shapes
: 
�
Ptraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/MulMulStraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/scalarKtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/truediv*
T0*B
_class8
64loc:@batch_normalization_2/moments/SquaredDifference*(
_output_shapes
:����������
�
Ptraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/subSubdense_2/Relu*batch_normalization_2/moments/StopGradientL^training/Adam/gradients/batch_normalization_2/moments/variance_grad/truediv*
T0*B
_class8
64loc:@batch_normalization_2/moments/SquaredDifference*(
_output_shapes
:����������
�
Rtraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/mul_1MulPtraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/MulPtraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/sub*
T0*B
_class8
64loc:@batch_normalization_2/moments/SquaredDifference*(
_output_shapes
:����������
�
Rtraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/ShapeShapedense_2/Relu*
T0*B
_class8
64loc:@batch_normalization_2/moments/SquaredDifference*
out_type0*
_output_shapes
:
�
Ttraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/Shape_1Shape*batch_normalization_2/moments/StopGradient*
T0*B
_class8
64loc:@batch_normalization_2/moments/SquaredDifference*
out_type0*
_output_shapes
:
�
btraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgsRtraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/ShapeTtraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/Shape_1*
T0*B
_class8
64loc:@batch_normalization_2/moments/SquaredDifference*2
_output_shapes 
:���������:���������
�
Ptraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/SumSumRtraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/mul_1btraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/BroadcastGradientArgs*
T0*B
_class8
64loc:@batch_normalization_2/moments/SquaredDifference*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Ttraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/ReshapeReshapePtraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/SumRtraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/Shape*
T0*B
_class8
64loc:@batch_normalization_2/moments/SquaredDifference*
Tshape0*(
_output_shapes
:����������
�
Rtraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/Sum_1SumRtraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/mul_1dtraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*B
_class8
64loc:@batch_normalization_2/moments/SquaredDifference*
_output_shapes
:
�
Vtraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/Reshape_1ReshapeRtraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/Sum_1Ttraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/Shape_1*
T0*B
_class8
64loc:@batch_normalization_2/moments/SquaredDifference*
Tshape0*
_output_shapes
:	�
�
Ptraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/NegNegVtraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/Reshape_1*
T0*B
_class8
64loc:@batch_normalization_2/moments/SquaredDifference*
_output_shapes
:	�
�
Etraining/Adam/gradients/batch_normalization_2/moments/mean_grad/ShapeShapedense_2/Relu*
_output_shapes
:*
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean*
out_type0
�
Dtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/SizeConst*5
_class+
)'loc:@batch_normalization_2/moments/mean*
value	B :*
dtype0*
_output_shapes
: 
�
Ctraining/Adam/gradients/batch_normalization_2/moments/mean_grad/addAddV24batch_normalization_2/moments/mean/reduction_indicesDtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Size*
_output_shapes
:*
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean
�
Ctraining/Adam/gradients/batch_normalization_2/moments/mean_grad/modFloorModCtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/addDtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Size*
_output_shapes
:*
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean
�
Gtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Shape_1Const*5
_class+
)'loc:@batch_normalization_2/moments/mean*
valueB:*
dtype0*
_output_shapes
:
�
Ktraining/Adam/gradients/batch_normalization_2/moments/mean_grad/range/startConst*
dtype0*
_output_shapes
: *5
_class+
)'loc:@batch_normalization_2/moments/mean*
value	B : 
�
Ktraining/Adam/gradients/batch_normalization_2/moments/mean_grad/range/deltaConst*5
_class+
)'loc:@batch_normalization_2/moments/mean*
value	B :*
dtype0*
_output_shapes
: 
�
Etraining/Adam/gradients/batch_normalization_2/moments/mean_grad/rangeRangeKtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/range/startDtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/SizeKtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/range/delta*

Tidx0*5
_class+
)'loc:@batch_normalization_2/moments/mean*
_output_shapes
:
�
Jtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Fill/valueConst*5
_class+
)'loc:@batch_normalization_2/moments/mean*
value	B :*
dtype0*
_output_shapes
: 
�
Dtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/FillFillGtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Shape_1Jtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Fill/value*
_output_shapes
:*
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean*

index_type0
�
Mtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/DynamicStitchDynamicStitchEtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/rangeCtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/modEtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/ShapeDtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Fill*
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean*
N*
_output_shapes
:
�
Itraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Maximum/yConst*
dtype0*
_output_shapes
: *5
_class+
)'loc:@batch_normalization_2/moments/mean*
value	B :
�
Gtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/MaximumMaximumMtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/DynamicStitchItraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Maximum/y*
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean*
_output_shapes
:
�
Htraining/Adam/gradients/batch_normalization_2/moments/mean_grad/floordivFloorDivEtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/ShapeGtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Maximum*
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean*
_output_shapes
:
�
Gtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/ReshapeReshapeJtraining/Adam/gradients/batch_normalization_2/moments/Squeeze_grad/ReshapeMtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/DynamicStitch*
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean*
Tshape0*0
_output_shapes
:������������������
�
Dtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/TileTileGtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/ReshapeHtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/floordiv*

Tmultiples0*
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean*0
_output_shapes
:������������������
�
Gtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Shape_2Shapedense_2/Relu*
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean*
out_type0*
_output_shapes
:
�
Gtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Shape_3Const*5
_class+
)'loc:@batch_normalization_2/moments/mean*
valueB"   �   *
dtype0*
_output_shapes
:
�
Etraining/Adam/gradients/batch_normalization_2/moments/mean_grad/ConstConst*5
_class+
)'loc:@batch_normalization_2/moments/mean*
valueB: *
dtype0*
_output_shapes
:
�
Dtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/ProdProdGtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Shape_2Etraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Const*
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean*
_output_shapes
: *
	keep_dims( *

Tidx0
�
Gtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Const_1Const*
dtype0*
_output_shapes
:*5
_class+
)'loc:@batch_normalization_2/moments/mean*
valueB: 
�
Ftraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Prod_1ProdGtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Shape_3Gtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Const_1*
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean*
_output_shapes
: *
	keep_dims( *

Tidx0
�
Ktraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Maximum_1/yConst*5
_class+
)'loc:@batch_normalization_2/moments/mean*
value	B :*
dtype0*
_output_shapes
: 
�
Itraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Maximum_1MaximumFtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Prod_1Ktraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Maximum_1/y*
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean*
_output_shapes
: 
�
Jtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/floordiv_1FloorDivDtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/ProdItraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Maximum_1*
_output_shapes
: *
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean
�
Dtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/CastCastJtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/floordiv_1*

SrcT0*5
_class+
)'loc:@batch_normalization_2/moments/mean*
Truncate( *

DstT0*
_output_shapes
: 
�
Gtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/truedivRealDivDtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/TileDtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Cast*
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean*(
_output_shapes
:����������
�
training/Adam/gradients/AddN_16AddNXtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1/Switch_grad/cond_gradJtraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/ReshapeTtraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/ReshapeGtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/truediv*
N*(
_output_shapes
:����������*
T0*
_class
loc:@dense_2/Relu
�
2training/Adam/gradients/dense_2/Relu_grad/ReluGradReluGradtraining/Adam/gradients/AddN_16dense_2/Relu*(
_output_shapes
:����������*
T0*
_class
loc:@dense_2/Relu
�
8training/Adam/gradients/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad2training/Adam/gradients/dense_2/Relu_grad/ReluGrad*
T0*"
_class
loc:@dense_2/BiasAdd*
data_formatNHWC*
_output_shapes	
:�
�
2training/Adam/gradients/dense_2/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_2/Relu_grad/ReluGraddense_2/MatMul/ReadVariableOp*
T0*!
_class
loc:@dense_2/MatMul*
transpose_a( *(
_output_shapes
:����������*
transpose_b(
�
4training/Adam/gradients/dense_2/MatMul_grad/MatMul_1MatMul batch_normalization_1/cond/Merge2training/Adam/gradients/dense_2/Relu_grad/ReluGrad*
transpose_b( *
T0*!
_class
loc:@dense_2/MatMul*
transpose_a(* 
_output_shapes
:
��
�
Gtraining/Adam/gradients/batch_normalization_1/cond/Merge_grad/cond_gradSwitch2training/Adam/gradients/dense_2/MatMul_grad/MatMul"batch_normalization_1/cond/pred_id*
T0*!
_class
loc:@dense_2/MatMul*<
_output_shapes*
(:����������:����������
�
Mtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/ShapeShape*batch_normalization_1/cond/batchnorm/mul_1*
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/add_1*
out_type0*
_output_shapes
:
�
Otraining/Adam/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/Shape_1Shape(batch_normalization_1/cond/batchnorm/sub*
_output_shapes
:*
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/add_1*
out_type0
�
]training/Adam/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsMtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/ShapeOtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/Shape_1*
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/add_1*2
_output_shapes 
:���������:���������
�
Ktraining/Adam/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/SumSumGtraining/Adam/gradients/batch_normalization_1/cond/Merge_grad/cond_grad]training/Adam/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/BroadcastGradientArgs*
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/add_1*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Otraining/Adam/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/ReshapeReshapeKtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/SumMtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/Shape*
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/add_1*
Tshape0*(
_output_shapes
:����������
�
Mtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/Sum_1SumGtraining/Adam/gradients/batch_normalization_1/cond/Merge_grad/cond_grad_training/Adam/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/add_1*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Qtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/Reshape_1ReshapeMtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/Sum_1Otraining/Adam/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/Shape_1*
_output_shapes	
:�*
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/add_1*
Tshape0
�
!training/Adam/gradients/Switch_16Switch%batch_normalization_1/batchnorm/add_1"batch_normalization_1/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1*<
_output_shapes*
(:����������:����������
�
#training/Adam/gradients/Identity_16Identity!training/Adam/gradients/Switch_16*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1*(
_output_shapes
:����������
�
 training/Adam/gradients/Shape_11Shape!training/Adam/gradients/Switch_16*
_output_shapes
:*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1*
out_type0
�
&training/Adam/gradients/zeros_16/ConstConst$^training/Adam/gradients/Identity_16*8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1*
valueB
 *    *
dtype0*
_output_shapes
: 
�
 training/Adam/gradients/zeros_16Fill training/Adam/gradients/Shape_11&training/Adam/gradients/zeros_16/Const*(
_output_shapes
:����������*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1*

index_type0
�
Jtraining/Adam/gradients/batch_normalization_1/cond/Switch_1_grad/cond_gradMerge training/Adam/gradients/zeros_16Itraining/Adam/gradients/batch_normalization_1/cond/Merge_grad/cond_grad:1*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1*
N**
_output_shapes
:����������: 
�
Mtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/ShapeShape1batch_normalization_1/cond/batchnorm/mul_1/Switch*
_output_shapes
:*
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/mul_1*
out_type0
�
Otraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/Shape_1Shape(batch_normalization_1/cond/batchnorm/mul*
_output_shapes
:*
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/mul_1*
out_type0
�
]training/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsMtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/ShapeOtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/Shape_1*
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/mul_1*2
_output_shapes 
:���������:���������
�
Ktraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/MulMulOtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/Reshape(batch_normalization_1/cond/batchnorm/mul*(
_output_shapes
:����������*
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/mul_1
�
Ktraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/SumSumKtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/Mul]training/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/mul_1*
_output_shapes
:
�
Otraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/ReshapeReshapeKtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/SumMtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/Shape*
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/mul_1*
Tshape0*(
_output_shapes
:����������
�
Mtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/Mul_1Mul1batch_normalization_1/cond/batchnorm/mul_1/SwitchOtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/Reshape*(
_output_shapes
:����������*
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/mul_1
�
Mtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/Sum_1SumMtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/Mul_1_training/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/mul_1*
_output_shapes
:
�
Qtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/Reshape_1ReshapeMtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/Sum_1Otraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/Shape_1*
_output_shapes	
:�*
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/mul_1*
Tshape0
�
Itraining/Adam/gradients/batch_normalization_1/cond/batchnorm/sub_grad/NegNegQtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:�*
T0*;
_class1
/-loc:@batch_normalization_1/cond/batchnorm/sub
�
Htraining/Adam/gradients/batch_normalization_1/batchnorm/add_1_grad/ShapeShape%batch_normalization_1/batchnorm/mul_1*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1*
out_type0*
_output_shapes
:
�
Jtraining/Adam/gradients/batch_normalization_1/batchnorm/add_1_grad/Shape_1Shape#batch_normalization_1/batchnorm/sub*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1*
out_type0*
_output_shapes
:
�
Xtraining/Adam/gradients/batch_normalization_1/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsHtraining/Adam/gradients/batch_normalization_1/batchnorm/add_1_grad/ShapeJtraining/Adam/gradients/batch_normalization_1/batchnorm/add_1_grad/Shape_1*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1*2
_output_shapes 
:���������:���������
�
Ftraining/Adam/gradients/batch_normalization_1/batchnorm/add_1_grad/SumSumJtraining/Adam/gradients/batch_normalization_1/cond/Switch_1_grad/cond_gradXtraining/Adam/gradients/batch_normalization_1/batchnorm/add_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1*
_output_shapes
:
�
Jtraining/Adam/gradients/batch_normalization_1/batchnorm/add_1_grad/ReshapeReshapeFtraining/Adam/gradients/batch_normalization_1/batchnorm/add_1_grad/SumHtraining/Adam/gradients/batch_normalization_1/batchnorm/add_1_grad/Shape*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1*
Tshape0*(
_output_shapes
:����������
�
Htraining/Adam/gradients/batch_normalization_1/batchnorm/add_1_grad/Sum_1SumJtraining/Adam/gradients/batch_normalization_1/cond/Switch_1_grad/cond_gradZtraining/Adam/gradients/batch_normalization_1/batchnorm/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1*
_output_shapes
:
�
Ltraining/Adam/gradients/batch_normalization_1/batchnorm/add_1_grad/Reshape_1ReshapeHtraining/Adam/gradients/batch_normalization_1/batchnorm/add_1_grad/Sum_1Jtraining/Adam/gradients/batch_normalization_1/batchnorm/add_1_grad/Shape_1*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1*
Tshape0*
_output_shapes	
:�
�
!training/Adam/gradients/Switch_17Switchdense_1/Relu"batch_normalization_1/cond/pred_id*<
_output_shapes*
(:����������:����������*
T0*
_class
loc:@dense_1/Relu
�
#training/Adam/gradients/Identity_17Identity#training/Adam/gradients/Switch_17:1*
T0*
_class
loc:@dense_1/Relu*(
_output_shapes
:����������
�
 training/Adam/gradients/Shape_12Shape#training/Adam/gradients/Switch_17:1*
_output_shapes
:*
T0*
_class
loc:@dense_1/Relu*
out_type0
�
&training/Adam/gradients/zeros_17/ConstConst$^training/Adam/gradients/Identity_17*
_class
loc:@dense_1/Relu*
valueB
 *    *
dtype0*
_output_shapes
: 
�
 training/Adam/gradients/zeros_17Fill training/Adam/gradients/Shape_12&training/Adam/gradients/zeros_17/Const*
T0*
_class
loc:@dense_1/Relu*

index_type0*(
_output_shapes
:����������
�
Xtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1/Switch_grad/cond_gradMergeOtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/Reshape training/Adam/gradients/zeros_17*
T0*
_class
loc:@dense_1/Relu*
N**
_output_shapes
:����������: 
�
Ktraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_2_grad/MulMulItraining/Adam/gradients/batch_normalization_1/cond/batchnorm/sub_grad/Neg(batch_normalization_1/cond/batchnorm/mul*
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/mul_2*
_output_shapes	
:�
�
Mtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_2_grad/Mul_1MulItraining/Adam/gradients/batch_normalization_1/cond/batchnorm/sub_grad/Neg5batch_normalization_1/cond/batchnorm/ReadVariableOp_1*
_output_shapes	
:�*
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/mul_2
�
Htraining/Adam/gradients/batch_normalization_1/batchnorm/mul_1_grad/ShapeShapedense_1/Relu*
_output_shapes
:*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/mul_1*
out_type0
�
Jtraining/Adam/gradients/batch_normalization_1/batchnorm/mul_1_grad/Shape_1Shape#batch_normalization_1/batchnorm/mul*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/mul_1*
out_type0*
_output_shapes
:
�
Xtraining/Adam/gradients/batch_normalization_1/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsHtraining/Adam/gradients/batch_normalization_1/batchnorm/mul_1_grad/ShapeJtraining/Adam/gradients/batch_normalization_1/batchnorm/mul_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/mul_1
�
Ftraining/Adam/gradients/batch_normalization_1/batchnorm/mul_1_grad/MulMulJtraining/Adam/gradients/batch_normalization_1/batchnorm/add_1_grad/Reshape#batch_normalization_1/batchnorm/mul*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/mul_1*(
_output_shapes
:����������
�
Ftraining/Adam/gradients/batch_normalization_1/batchnorm/mul_1_grad/SumSumFtraining/Adam/gradients/batch_normalization_1/batchnorm/mul_1_grad/MulXtraining/Adam/gradients/batch_normalization_1/batchnorm/mul_1_grad/BroadcastGradientArgs*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/mul_1*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Jtraining/Adam/gradients/batch_normalization_1/batchnorm/mul_1_grad/ReshapeReshapeFtraining/Adam/gradients/batch_normalization_1/batchnorm/mul_1_grad/SumHtraining/Adam/gradients/batch_normalization_1/batchnorm/mul_1_grad/Shape*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/mul_1*
Tshape0*(
_output_shapes
:����������
�
Htraining/Adam/gradients/batch_normalization_1/batchnorm/mul_1_grad/Mul_1Muldense_1/ReluJtraining/Adam/gradients/batch_normalization_1/batchnorm/add_1_grad/Reshape*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/mul_1*(
_output_shapes
:����������
�
Htraining/Adam/gradients/batch_normalization_1/batchnorm/mul_1_grad/Sum_1SumHtraining/Adam/gradients/batch_normalization_1/batchnorm/mul_1_grad/Mul_1Ztraining/Adam/gradients/batch_normalization_1/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/mul_1*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Ltraining/Adam/gradients/batch_normalization_1/batchnorm/mul_1_grad/Reshape_1ReshapeHtraining/Adam/gradients/batch_normalization_1/batchnorm/mul_1_grad/Sum_1Jtraining/Adam/gradients/batch_normalization_1/batchnorm/mul_1_grad/Shape_1*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/mul_1*
Tshape0*
_output_shapes	
:�
�
Dtraining/Adam/gradients/batch_normalization_1/batchnorm/sub_grad/NegNegLtraining/Adam/gradients/batch_normalization_1/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:�*
T0*6
_class,
*(loc:@batch_normalization_1/batchnorm/sub
�
!training/Adam/gradients/Switch_18Switchbatch_normalization_1/beta"batch_normalization_1/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_1/beta*
_output_shapes
: : 
�
#training/Adam/gradients/Identity_18Identity#training/Adam/gradients/Switch_18:1*
_output_shapes
: *
T0*-
_class#
!loc:@batch_normalization_1/beta
�
'training/Adam/gradients/VariableShape_6VariableShape#training/Adam/gradients/Switch_18:1$^training/Adam/gradients/Identity_18*-
_class#
!loc:@batch_normalization_1/beta*
out_type0*
_output_shapes
:
�
&training/Adam/gradients/zeros_18/ConstConst$^training/Adam/gradients/Identity_18*-
_class#
!loc:@batch_normalization_1/beta*
valueB
 *    *
dtype0*
_output_shapes
: 
�
 training/Adam/gradients/zeros_18Fill'training/Adam/gradients/VariableShape_6&training/Adam/gradients/zeros_18/Const*
T0*-
_class#
!loc:@batch_normalization_1/beta*

index_type0*#
_output_shapes
:���������
�
ctraining/Adam/gradients/batch_normalization_1/cond/batchnorm/ReadVariableOp_2/Switch_grad/cond_gradMergeQtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/Reshape_1 training/Adam/gradients/zeros_18*
T0*-
_class#
!loc:@batch_normalization_1/beta*
N*%
_output_shapes
:���������: 
�
training/Adam/gradients/AddN_17AddNQtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/Reshape_1Mtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_2_grad/Mul_1*
N*
_output_shapes	
:�*
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/mul_1
�
Itraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_grad/MulMultraining/Adam/gradients/AddN_177batch_normalization_1/cond/batchnorm/mul/ReadVariableOp*
_output_shapes	
:�*
T0*;
_class1
/-loc:@batch_normalization_1/cond/batchnorm/mul
�
Ktraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_grad/Mul_1Multraining/Adam/gradients/AddN_17*batch_normalization_1/cond/batchnorm/Rsqrt*
T0*;
_class1
/-loc:@batch_normalization_1/cond/batchnorm/mul*
_output_shapes	
:�
�
Ftraining/Adam/gradients/batch_normalization_1/batchnorm/mul_2_grad/MulMulDtraining/Adam/gradients/batch_normalization_1/batchnorm/sub_grad/Neg#batch_normalization_1/batchnorm/mul*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/mul_2*
_output_shapes	
:�
�
Htraining/Adam/gradients/batch_normalization_1/batchnorm/mul_2_grad/Mul_1MulDtraining/Adam/gradients/batch_normalization_1/batchnorm/sub_grad/Neg%batch_normalization_1/moments/Squeeze*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/mul_2*
_output_shapes	
:�
�
training/Adam/gradients/AddN_18AddNctraining/Adam/gradients/batch_normalization_1/cond/batchnorm/ReadVariableOp_2/Switch_grad/cond_gradLtraining/Adam/gradients/batch_normalization_1/batchnorm/add_1_grad/Reshape_1*
N*
_output_shapes	
:�*
T0*-
_class#
!loc:@batch_normalization_1/beta
�
Htraining/Adam/gradients/batch_normalization_1/moments/Squeeze_grad/ShapeConst*8
_class.
,*loc:@batch_normalization_1/moments/Squeeze*
valueB"   �   *
dtype0*
_output_shapes
:
�
Jtraining/Adam/gradients/batch_normalization_1/moments/Squeeze_grad/ReshapeReshapeFtraining/Adam/gradients/batch_normalization_1/batchnorm/mul_2_grad/MulHtraining/Adam/gradients/batch_normalization_1/moments/Squeeze_grad/Shape*
T0*8
_class.
,*loc:@batch_normalization_1/moments/Squeeze*
Tshape0*
_output_shapes
:	�
�
training/Adam/gradients/AddN_19AddNLtraining/Adam/gradients/batch_normalization_1/batchnorm/mul_1_grad/Reshape_1Htraining/Adam/gradients/batch_normalization_1/batchnorm/mul_2_grad/Mul_1*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/mul_1*
N*
_output_shapes	
:�
�
Dtraining/Adam/gradients/batch_normalization_1/batchnorm/mul_grad/MulMultraining/Adam/gradients/AddN_192batch_normalization_1/batchnorm/mul/ReadVariableOp*
T0*6
_class,
*(loc:@batch_normalization_1/batchnorm/mul*
_output_shapes	
:�
�
Ftraining/Adam/gradients/batch_normalization_1/batchnorm/mul_grad/Mul_1Multraining/Adam/gradients/AddN_19%batch_normalization_1/batchnorm/Rsqrt*
T0*6
_class,
*(loc:@batch_normalization_1/batchnorm/mul*
_output_shapes	
:�
�
!training/Adam/gradients/Switch_19Switchbatch_normalization_1/gamma"batch_normalization_1/cond/pred_id*
_output_shapes
: : *
T0*.
_class$
" loc:@batch_normalization_1/gamma
�
#training/Adam/gradients/Identity_19Identity#training/Adam/gradients/Switch_19:1*
T0*.
_class$
" loc:@batch_normalization_1/gamma*
_output_shapes
: 
�
'training/Adam/gradients/VariableShape_7VariableShape#training/Adam/gradients/Switch_19:1$^training/Adam/gradients/Identity_19*.
_class$
" loc:@batch_normalization_1/gamma*
out_type0*
_output_shapes
:
�
&training/Adam/gradients/zeros_19/ConstConst$^training/Adam/gradients/Identity_19*
dtype0*
_output_shapes
: *.
_class$
" loc:@batch_normalization_1/gamma*
valueB
 *    
�
 training/Adam/gradients/zeros_19Fill'training/Adam/gradients/VariableShape_7&training/Adam/gradients/zeros_19/Const*
T0*.
_class$
" loc:@batch_normalization_1/gamma*

index_type0*#
_output_shapes
:���������
�
etraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul/ReadVariableOp/Switch_grad/cond_gradMergeKtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_grad/Mul_1 training/Adam/gradients/zeros_19*
T0*.
_class$
" loc:@batch_normalization_1/gamma*
N*%
_output_shapes
:���������: 
�
Ltraining/Adam/gradients/batch_normalization_1/batchnorm/Rsqrt_grad/RsqrtGrad	RsqrtGrad%batch_normalization_1/batchnorm/RsqrtDtraining/Adam/gradients/batch_normalization_1/batchnorm/mul_grad/Mul*
_output_shapes	
:�*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/Rsqrt
�
Vtraining/Adam/gradients/batch_normalization_1/batchnorm/add_grad/Sum/reduction_indicesConst*6
_class,
*(loc:@batch_normalization_1/batchnorm/add*
valueB: *
dtype0*
_output_shapes
:
�
Dtraining/Adam/gradients/batch_normalization_1/batchnorm/add_grad/SumSumLtraining/Adam/gradients/batch_normalization_1/batchnorm/Rsqrt_grad/RsqrtGradVtraining/Adam/gradients/batch_normalization_1/batchnorm/add_grad/Sum/reduction_indices*
T0*6
_class,
*(loc:@batch_normalization_1/batchnorm/add*
_output_shapes
: *
	keep_dims( *

Tidx0
�
Ntraining/Adam/gradients/batch_normalization_1/batchnorm/add_grad/Reshape/shapeConst*
dtype0*
_output_shapes
: *6
_class,
*(loc:@batch_normalization_1/batchnorm/add*
valueB 
�
Htraining/Adam/gradients/batch_normalization_1/batchnorm/add_grad/ReshapeReshapeDtraining/Adam/gradients/batch_normalization_1/batchnorm/add_grad/SumNtraining/Adam/gradients/batch_normalization_1/batchnorm/add_grad/Reshape/shape*
T0*6
_class,
*(loc:@batch_normalization_1/batchnorm/add*
Tshape0*
_output_shapes
: 
�
training/Adam/gradients/AddN_20AddNetraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul/ReadVariableOp/Switch_grad/cond_gradFtraining/Adam/gradients/batch_normalization_1/batchnorm/mul_grad/Mul_1*
N*
_output_shapes	
:�*
T0*.
_class$
" loc:@batch_normalization_1/gamma
�
Jtraining/Adam/gradients/batch_normalization_1/moments/Squeeze_1_grad/ShapeConst*:
_class0
.,loc:@batch_normalization_1/moments/Squeeze_1*
valueB"   �   *
dtype0*
_output_shapes
:
�
Ltraining/Adam/gradients/batch_normalization_1/moments/Squeeze_1_grad/ReshapeReshapeLtraining/Adam/gradients/batch_normalization_1/batchnorm/Rsqrt_grad/RsqrtGradJtraining/Adam/gradients/batch_normalization_1/moments/Squeeze_1_grad/Shape*
T0*:
_class0
.,loc:@batch_normalization_1/moments/Squeeze_1*
Tshape0*
_output_shapes
:	�
�
Itraining/Adam/gradients/batch_normalization_1/moments/variance_grad/ShapeShape/batch_normalization_1/moments/SquaredDifference*
T0*9
_class/
-+loc:@batch_normalization_1/moments/variance*
out_type0*
_output_shapes
:
�
Htraining/Adam/gradients/batch_normalization_1/moments/variance_grad/SizeConst*9
_class/
-+loc:@batch_normalization_1/moments/variance*
value	B :*
dtype0*
_output_shapes
: 
�
Gtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/addAddV28batch_normalization_1/moments/variance/reduction_indicesHtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Size*
T0*9
_class/
-+loc:@batch_normalization_1/moments/variance*
_output_shapes
:
�
Gtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/modFloorModGtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/addHtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Size*
T0*9
_class/
-+loc:@batch_normalization_1/moments/variance*
_output_shapes
:
�
Ktraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Shape_1Const*9
_class/
-+loc:@batch_normalization_1/moments/variance*
valueB:*
dtype0*
_output_shapes
:
�
Otraining/Adam/gradients/batch_normalization_1/moments/variance_grad/range/startConst*
dtype0*
_output_shapes
: *9
_class/
-+loc:@batch_normalization_1/moments/variance*
value	B : 
�
Otraining/Adam/gradients/batch_normalization_1/moments/variance_grad/range/deltaConst*9
_class/
-+loc:@batch_normalization_1/moments/variance*
value	B :*
dtype0*
_output_shapes
: 
�
Itraining/Adam/gradients/batch_normalization_1/moments/variance_grad/rangeRangeOtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/range/startHtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/SizeOtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/range/delta*9
_class/
-+loc:@batch_normalization_1/moments/variance*
_output_shapes
:*

Tidx0
�
Ntraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Fill/valueConst*9
_class/
-+loc:@batch_normalization_1/moments/variance*
value	B :*
dtype0*
_output_shapes
: 
�
Htraining/Adam/gradients/batch_normalization_1/moments/variance_grad/FillFillKtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Shape_1Ntraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Fill/value*
T0*9
_class/
-+loc:@batch_normalization_1/moments/variance*

index_type0*
_output_shapes
:
�
Qtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/DynamicStitchDynamicStitchItraining/Adam/gradients/batch_normalization_1/moments/variance_grad/rangeGtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/modItraining/Adam/gradients/batch_normalization_1/moments/variance_grad/ShapeHtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Fill*
T0*9
_class/
-+loc:@batch_normalization_1/moments/variance*
N*
_output_shapes
:
�
Mtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Maximum/yConst*
dtype0*
_output_shapes
: *9
_class/
-+loc:@batch_normalization_1/moments/variance*
value	B :
�
Ktraining/Adam/gradients/batch_normalization_1/moments/variance_grad/MaximumMaximumQtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/DynamicStitchMtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Maximum/y*
T0*9
_class/
-+loc:@batch_normalization_1/moments/variance*
_output_shapes
:
�
Ltraining/Adam/gradients/batch_normalization_1/moments/variance_grad/floordivFloorDivItraining/Adam/gradients/batch_normalization_1/moments/variance_grad/ShapeKtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Maximum*
T0*9
_class/
-+loc:@batch_normalization_1/moments/variance*
_output_shapes
:
�
Ktraining/Adam/gradients/batch_normalization_1/moments/variance_grad/ReshapeReshapeLtraining/Adam/gradients/batch_normalization_1/moments/Squeeze_1_grad/ReshapeQtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/DynamicStitch*
T0*9
_class/
-+loc:@batch_normalization_1/moments/variance*
Tshape0*0
_output_shapes
:������������������
�
Htraining/Adam/gradients/batch_normalization_1/moments/variance_grad/TileTileKtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/ReshapeLtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/floordiv*0
_output_shapes
:������������������*

Tmultiples0*
T0*9
_class/
-+loc:@batch_normalization_1/moments/variance
�
Ktraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Shape_2Shape/batch_normalization_1/moments/SquaredDifference*
T0*9
_class/
-+loc:@batch_normalization_1/moments/variance*
out_type0*
_output_shapes
:
�
Ktraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Shape_3Const*9
_class/
-+loc:@batch_normalization_1/moments/variance*
valueB"   �   *
dtype0*
_output_shapes
:
�
Itraining/Adam/gradients/batch_normalization_1/moments/variance_grad/ConstConst*
dtype0*
_output_shapes
:*9
_class/
-+loc:@batch_normalization_1/moments/variance*
valueB: 
�
Htraining/Adam/gradients/batch_normalization_1/moments/variance_grad/ProdProdKtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Shape_2Itraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0*9
_class/
-+loc:@batch_normalization_1/moments/variance
�
Ktraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Const_1Const*
dtype0*
_output_shapes
:*9
_class/
-+loc:@batch_normalization_1/moments/variance*
valueB: 
�
Jtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Prod_1ProdKtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Shape_3Ktraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Const_1*
T0*9
_class/
-+loc:@batch_normalization_1/moments/variance*
_output_shapes
: *
	keep_dims( *

Tidx0
�
Otraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Maximum_1/yConst*
dtype0*
_output_shapes
: *9
_class/
-+loc:@batch_normalization_1/moments/variance*
value	B :
�
Mtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Maximum_1MaximumJtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Prod_1Otraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Maximum_1/y*
T0*9
_class/
-+loc:@batch_normalization_1/moments/variance*
_output_shapes
: 
�
Ntraining/Adam/gradients/batch_normalization_1/moments/variance_grad/floordiv_1FloorDivHtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/ProdMtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Maximum_1*
T0*9
_class/
-+loc:@batch_normalization_1/moments/variance*
_output_shapes
: 
�
Htraining/Adam/gradients/batch_normalization_1/moments/variance_grad/CastCastNtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/floordiv_1*

SrcT0*9
_class/
-+loc:@batch_normalization_1/moments/variance*
Truncate( *

DstT0*
_output_shapes
: 
�
Ktraining/Adam/gradients/batch_normalization_1/moments/variance_grad/truedivRealDivHtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/TileHtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Cast*(
_output_shapes
:����������*
T0*9
_class/
-+loc:@batch_normalization_1/moments/variance
�
Straining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/scalarConstL^training/Adam/gradients/batch_normalization_1/moments/variance_grad/truediv*B
_class8
64loc:@batch_normalization_1/moments/SquaredDifference*
valueB
 *   @*
dtype0*
_output_shapes
: 
�
Ptraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/MulMulStraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/scalarKtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/truediv*(
_output_shapes
:����������*
T0*B
_class8
64loc:@batch_normalization_1/moments/SquaredDifference
�
Ptraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/subSubdense_1/Relu*batch_normalization_1/moments/StopGradientL^training/Adam/gradients/batch_normalization_1/moments/variance_grad/truediv*
T0*B
_class8
64loc:@batch_normalization_1/moments/SquaredDifference*(
_output_shapes
:����������
�
Rtraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/mul_1MulPtraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/MulPtraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/sub*
T0*B
_class8
64loc:@batch_normalization_1/moments/SquaredDifference*(
_output_shapes
:����������
�
Rtraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/ShapeShapedense_1/Relu*
T0*B
_class8
64loc:@batch_normalization_1/moments/SquaredDifference*
out_type0*
_output_shapes
:
�
Ttraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/Shape_1Shape*batch_normalization_1/moments/StopGradient*
_output_shapes
:*
T0*B
_class8
64loc:@batch_normalization_1/moments/SquaredDifference*
out_type0
�
btraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgsRtraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/ShapeTtraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/Shape_1*
T0*B
_class8
64loc:@batch_normalization_1/moments/SquaredDifference*2
_output_shapes 
:���������:���������
�
Ptraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/SumSumRtraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/mul_1btraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/BroadcastGradientArgs*
T0*B
_class8
64loc:@batch_normalization_1/moments/SquaredDifference*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Ttraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/ReshapeReshapePtraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/SumRtraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/Shape*
T0*B
_class8
64loc:@batch_normalization_1/moments/SquaredDifference*
Tshape0*(
_output_shapes
:����������
�
Rtraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/Sum_1SumRtraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/mul_1dtraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*B
_class8
64loc:@batch_normalization_1/moments/SquaredDifference*
_output_shapes
:
�
Vtraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/Reshape_1ReshapeRtraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/Sum_1Ttraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/Shape_1*
_output_shapes
:	�*
T0*B
_class8
64loc:@batch_normalization_1/moments/SquaredDifference*
Tshape0
�
Ptraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/NegNegVtraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/Reshape_1*
T0*B
_class8
64loc:@batch_normalization_1/moments/SquaredDifference*
_output_shapes
:	�
�
Etraining/Adam/gradients/batch_normalization_1/moments/mean_grad/ShapeShapedense_1/Relu*
T0*5
_class+
)'loc:@batch_normalization_1/moments/mean*
out_type0*
_output_shapes
:
�
Dtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/SizeConst*5
_class+
)'loc:@batch_normalization_1/moments/mean*
value	B :*
dtype0*
_output_shapes
: 
�
Ctraining/Adam/gradients/batch_normalization_1/moments/mean_grad/addAddV24batch_normalization_1/moments/mean/reduction_indicesDtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Size*
_output_shapes
:*
T0*5
_class+
)'loc:@batch_normalization_1/moments/mean
�
Ctraining/Adam/gradients/batch_normalization_1/moments/mean_grad/modFloorModCtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/addDtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Size*
T0*5
_class+
)'loc:@batch_normalization_1/moments/mean*
_output_shapes
:
�
Gtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Shape_1Const*5
_class+
)'loc:@batch_normalization_1/moments/mean*
valueB:*
dtype0*
_output_shapes
:
�
Ktraining/Adam/gradients/batch_normalization_1/moments/mean_grad/range/startConst*5
_class+
)'loc:@batch_normalization_1/moments/mean*
value	B : *
dtype0*
_output_shapes
: 
�
Ktraining/Adam/gradients/batch_normalization_1/moments/mean_grad/range/deltaConst*
dtype0*
_output_shapes
: *5
_class+
)'loc:@batch_normalization_1/moments/mean*
value	B :
�
Etraining/Adam/gradients/batch_normalization_1/moments/mean_grad/rangeRangeKtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/range/startDtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/SizeKtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/range/delta*5
_class+
)'loc:@batch_normalization_1/moments/mean*
_output_shapes
:*

Tidx0
�
Jtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Fill/valueConst*
dtype0*
_output_shapes
: *5
_class+
)'loc:@batch_normalization_1/moments/mean*
value	B :
�
Dtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/FillFillGtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Shape_1Jtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Fill/value*
_output_shapes
:*
T0*5
_class+
)'loc:@batch_normalization_1/moments/mean*

index_type0
�
Mtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/DynamicStitchDynamicStitchEtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/rangeCtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/modEtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/ShapeDtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Fill*
T0*5
_class+
)'loc:@batch_normalization_1/moments/mean*
N*
_output_shapes
:
�
Itraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Maximum/yConst*5
_class+
)'loc:@batch_normalization_1/moments/mean*
value	B :*
dtype0*
_output_shapes
: 
�
Gtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/MaximumMaximumMtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/DynamicStitchItraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Maximum/y*
T0*5
_class+
)'loc:@batch_normalization_1/moments/mean*
_output_shapes
:
�
Htraining/Adam/gradients/batch_normalization_1/moments/mean_grad/floordivFloorDivEtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/ShapeGtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Maximum*
T0*5
_class+
)'loc:@batch_normalization_1/moments/mean*
_output_shapes
:
�
Gtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/ReshapeReshapeJtraining/Adam/gradients/batch_normalization_1/moments/Squeeze_grad/ReshapeMtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/DynamicStitch*
T0*5
_class+
)'loc:@batch_normalization_1/moments/mean*
Tshape0*0
_output_shapes
:������������������
�
Dtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/TileTileGtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/ReshapeHtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/floordiv*

Tmultiples0*
T0*5
_class+
)'loc:@batch_normalization_1/moments/mean*0
_output_shapes
:������������������
�
Gtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Shape_2Shapedense_1/Relu*
_output_shapes
:*
T0*5
_class+
)'loc:@batch_normalization_1/moments/mean*
out_type0
�
Gtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Shape_3Const*5
_class+
)'loc:@batch_normalization_1/moments/mean*
valueB"   �   *
dtype0*
_output_shapes
:
�
Etraining/Adam/gradients/batch_normalization_1/moments/mean_grad/ConstConst*
dtype0*
_output_shapes
:*5
_class+
)'loc:@batch_normalization_1/moments/mean*
valueB: 
�
Dtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/ProdProdGtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Shape_2Etraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0*5
_class+
)'loc:@batch_normalization_1/moments/mean
�
Gtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Const_1Const*5
_class+
)'loc:@batch_normalization_1/moments/mean*
valueB: *
dtype0*
_output_shapes
:
�
Ftraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Prod_1ProdGtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Shape_3Gtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Const_1*
T0*5
_class+
)'loc:@batch_normalization_1/moments/mean*
_output_shapes
: *
	keep_dims( *

Tidx0
�
Ktraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Maximum_1/yConst*5
_class+
)'loc:@batch_normalization_1/moments/mean*
value	B :*
dtype0*
_output_shapes
: 
�
Itraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Maximum_1MaximumFtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Prod_1Ktraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Maximum_1/y*
T0*5
_class+
)'loc:@batch_normalization_1/moments/mean*
_output_shapes
: 
�
Jtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/floordiv_1FloorDivDtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/ProdItraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Maximum_1*
T0*5
_class+
)'loc:@batch_normalization_1/moments/mean*
_output_shapes
: 
�
Dtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/CastCastJtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/floordiv_1*

SrcT0*5
_class+
)'loc:@batch_normalization_1/moments/mean*
Truncate( *

DstT0*
_output_shapes
: 
�
Gtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/truedivRealDivDtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/TileDtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Cast*(
_output_shapes
:����������*
T0*5
_class+
)'loc:@batch_normalization_1/moments/mean
�
training/Adam/gradients/AddN_21AddNXtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1/Switch_grad/cond_gradJtraining/Adam/gradients/batch_normalization_1/batchnorm/mul_1_grad/ReshapeTtraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/ReshapeGtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/truediv*
T0*
_class
loc:@dense_1/Relu*
N*(
_output_shapes
:����������
�
2training/Adam/gradients/dense_1/Relu_grad/ReluGradReluGradtraining/Adam/gradients/AddN_21dense_1/Relu*
T0*
_class
loc:@dense_1/Relu*(
_output_shapes
:����������
�
8training/Adam/gradients/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad2training/Adam/gradients/dense_1/Relu_grad/ReluGrad*
T0*"
_class
loc:@dense_1/BiasAdd*
data_formatNHWC*
_output_shapes	
:�
�
2training/Adam/gradients/dense_1/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_1/Relu_grad/ReluGraddense_1/MatMul/ReadVariableOp*
T0*!
_class
loc:@dense_1/MatMul*
transpose_a( *(
_output_shapes
:����������*
transpose_b(
�
4training/Adam/gradients/dense_1/MatMul_grad/MatMul_1MatMuldense_1_input2training/Adam/gradients/dense_1/Relu_grad/ReluGrad*
T0*!
_class
loc:@dense_1/MatMul*
transpose_a(* 
_output_shapes
:
��*
transpose_b( 
U
training/Adam/ConstConst*
value	B	 R*
dtype0	*
_output_shapes
: 
k
!training/Adam/AssignAddVariableOpAssignAddVariableOpAdam/iterationstraining/Adam/Const*
dtype0	
�
training/Adam/ReadVariableOpReadVariableOpAdam/iterations"^training/Adam/AssignAddVariableOp*
dtype0	*
_output_shapes
: 
i
!training/Adam/Cast/ReadVariableOpReadVariableOpAdam/iterations*
dtype0	*
_output_shapes
: 
}
training/Adam/CastCast!training/Adam/Cast/ReadVariableOp*

SrcT0	*
Truncate( *

DstT0*
_output_shapes
: 
X
training/Adam/add/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
d
training/Adam/addAddV2training/Adam/Casttraining/Adam/add/y*
T0*
_output_shapes
: 
d
 training/Adam/Pow/ReadVariableOpReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
n
training/Adam/PowPow training/Adam/Pow/ReadVariableOptraining/Adam/add*
T0*
_output_shapes
: 
X
training/Adam/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
a
training/Adam/subSubtraining/Adam/sub/xtraining/Adam/Pow*
T0*
_output_shapes
: 
Z
training/Adam/Const_1Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_2Const*
dtype0*
_output_shapes
: *
valueB
 *  �
y
#training/Adam/clip_by_value/MinimumMinimumtraining/Adam/subtraining/Adam/Const_2*
T0*
_output_shapes
: 
�
training/Adam/clip_by_valueMaximum#training/Adam/clip_by_value/Minimumtraining/Adam/Const_1*
T0*
_output_shapes
: 
X
training/Adam/SqrtSqrttraining/Adam/clip_by_value*
_output_shapes
: *
T0
f
"training/Adam/Pow_1/ReadVariableOpReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
r
training/Adam/Pow_1Pow"training/Adam/Pow_1/ReadVariableOptraining/Adam/add*
T0*
_output_shapes
: 
Z
training/Adam/sub_1/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
g
training/Adam/sub_1Subtraining/Adam/sub_1/xtraining/Adam/Pow_1*
T0*
_output_shapes
: 
j
training/Adam/truedivRealDivtraining/Adam/Sqrttraining/Adam/sub_1*
T0*
_output_shapes
: 
i
training/Adam/ReadVariableOp_1ReadVariableOpAdam/learning_rate*
dtype0*
_output_shapes
: 
p
training/Adam/mulMultraining/Adam/ReadVariableOp_1training/Adam/truediv*
T0*
_output_shapes
: 
r
!training/Adam/m_0/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB"   �   
\
training/Adam/m_0/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/m_0Fill!training/Adam/m_0/shape_as_tensortraining/Adam/m_0/Const*
T0*

index_type0* 
_output_shapes
:
��
�
training/Adam/m_0_1VarHandleOp*
shape:
��*
dtype0*
_output_shapes
: *$
shared_nametraining/Adam/m_0_1*&
_class
loc:@training/Adam/m_0_1*
	container 
w
4training/Adam/m_0_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/m_0_1*
_output_shapes
: 
c
training/Adam/m_0_1/AssignAssignVariableOptraining/Adam/m_0_1training/Adam/m_0*
dtype0
}
'training/Adam/m_0_1/Read/ReadVariableOpReadVariableOptraining/Adam/m_0_1*
dtype0* 
_output_shapes
:
��
`
training/Adam/m_1Const*
valueB�*    *
dtype0*
_output_shapes	
:�
�
training/Adam/m_1_1VarHandleOp*
dtype0*
_output_shapes
: *$
shared_nametraining/Adam/m_1_1*&
_class
loc:@training/Adam/m_1_1*
	container *
shape:�
w
4training/Adam/m_1_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/m_1_1*
_output_shapes
: 
c
training/Adam/m_1_1/AssignAssignVariableOptraining/Adam/m_1_1training/Adam/m_1*
dtype0
x
'training/Adam/m_1_1/Read/ReadVariableOpReadVariableOptraining/Adam/m_1_1*
dtype0*
_output_shapes	
:�
`
training/Adam/m_2Const*
valueB�*    *
dtype0*
_output_shapes	
:�
�
training/Adam/m_2_1VarHandleOp*
shape:�*
dtype0*
_output_shapes
: *$
shared_nametraining/Adam/m_2_1*&
_class
loc:@training/Adam/m_2_1*
	container 
w
4training/Adam/m_2_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/m_2_1*
_output_shapes
: 
c
training/Adam/m_2_1/AssignAssignVariableOptraining/Adam/m_2_1training/Adam/m_2*
dtype0
x
'training/Adam/m_2_1/Read/ReadVariableOpReadVariableOptraining/Adam/m_2_1*
dtype0*
_output_shapes	
:�
`
training/Adam/m_3Const*
valueB�*    *
dtype0*
_output_shapes	
:�
�
training/Adam/m_3_1VarHandleOp*
dtype0*
_output_shapes
: *$
shared_nametraining/Adam/m_3_1*&
_class
loc:@training/Adam/m_3_1*
	container *
shape:�
w
4training/Adam/m_3_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/m_3_1*
_output_shapes
: 
c
training/Adam/m_3_1/AssignAssignVariableOptraining/Adam/m_3_1training/Adam/m_3*
dtype0
x
'training/Adam/m_3_1/Read/ReadVariableOpReadVariableOptraining/Adam/m_3_1*
dtype0*
_output_shapes	
:�
r
!training/Adam/m_4/shape_as_tensorConst*
valueB"�   �   *
dtype0*
_output_shapes
:
\
training/Adam/m_4/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/m_4Fill!training/Adam/m_4/shape_as_tensortraining/Adam/m_4/Const*
T0*

index_type0* 
_output_shapes
:
��
�
training/Adam/m_4_1VarHandleOp*
dtype0*
_output_shapes
: *$
shared_nametraining/Adam/m_4_1*&
_class
loc:@training/Adam/m_4_1*
	container *
shape:
��
w
4training/Adam/m_4_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/m_4_1*
_output_shapes
: 
c
training/Adam/m_4_1/AssignAssignVariableOptraining/Adam/m_4_1training/Adam/m_4*
dtype0
}
'training/Adam/m_4_1/Read/ReadVariableOpReadVariableOptraining/Adam/m_4_1*
dtype0* 
_output_shapes
:
��
`
training/Adam/m_5Const*
valueB�*    *
dtype0*
_output_shapes	
:�
�
training/Adam/m_5_1VarHandleOp*
dtype0*
_output_shapes
: *$
shared_nametraining/Adam/m_5_1*&
_class
loc:@training/Adam/m_5_1*
	container *
shape:�
w
4training/Adam/m_5_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/m_5_1*
_output_shapes
: 
c
training/Adam/m_5_1/AssignAssignVariableOptraining/Adam/m_5_1training/Adam/m_5*
dtype0
x
'training/Adam/m_5_1/Read/ReadVariableOpReadVariableOptraining/Adam/m_5_1*
dtype0*
_output_shapes	
:�
`
training/Adam/m_6Const*
dtype0*
_output_shapes	
:�*
valueB�*    
�
training/Adam/m_6_1VarHandleOp*
dtype0*
_output_shapes
: *$
shared_nametraining/Adam/m_6_1*&
_class
loc:@training/Adam/m_6_1*
	container *
shape:�
w
4training/Adam/m_6_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/m_6_1*
_output_shapes
: 
c
training/Adam/m_6_1/AssignAssignVariableOptraining/Adam/m_6_1training/Adam/m_6*
dtype0
x
'training/Adam/m_6_1/Read/ReadVariableOpReadVariableOptraining/Adam/m_6_1*
dtype0*
_output_shapes	
:�
`
training/Adam/m_7Const*
dtype0*
_output_shapes	
:�*
valueB�*    
�
training/Adam/m_7_1VarHandleOp*
dtype0*
_output_shapes
: *$
shared_nametraining/Adam/m_7_1*&
_class
loc:@training/Adam/m_7_1*
	container *
shape:�
w
4training/Adam/m_7_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/m_7_1*
_output_shapes
: 
c
training/Adam/m_7_1/AssignAssignVariableOptraining/Adam/m_7_1training/Adam/m_7*
dtype0
x
'training/Adam/m_7_1/Read/ReadVariableOpReadVariableOptraining/Adam/m_7_1*
dtype0*
_output_shapes	
:�
r
!training/Adam/m_8/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB"�   �   
\
training/Adam/m_8/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
�
training/Adam/m_8Fill!training/Adam/m_8/shape_as_tensortraining/Adam/m_8/Const*
T0*

index_type0* 
_output_shapes
:
��
�
training/Adam/m_8_1VarHandleOp*
dtype0*
_output_shapes
: *$
shared_nametraining/Adam/m_8_1*&
_class
loc:@training/Adam/m_8_1*
	container *
shape:
��
w
4training/Adam/m_8_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/m_8_1*
_output_shapes
: 
c
training/Adam/m_8_1/AssignAssignVariableOptraining/Adam/m_8_1training/Adam/m_8*
dtype0
}
'training/Adam/m_8_1/Read/ReadVariableOpReadVariableOptraining/Adam/m_8_1*
dtype0* 
_output_shapes
:
��
`
training/Adam/m_9Const*
valueB�*    *
dtype0*
_output_shapes	
:�
�
training/Adam/m_9_1VarHandleOp*
shape:�*
dtype0*
_output_shapes
: *$
shared_nametraining/Adam/m_9_1*&
_class
loc:@training/Adam/m_9_1*
	container 
w
4training/Adam/m_9_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/m_9_1*
_output_shapes
: 
c
training/Adam/m_9_1/AssignAssignVariableOptraining/Adam/m_9_1training/Adam/m_9*
dtype0
x
'training/Adam/m_9_1/Read/ReadVariableOpReadVariableOptraining/Adam/m_9_1*
dtype0*
_output_shapes	
:�
a
training/Adam/m_10Const*
valueB�*    *
dtype0*
_output_shapes	
:�
�
training/Adam/m_10_1VarHandleOp*'
_class
loc:@training/Adam/m_10_1*
	container *
shape:�*
dtype0*
_output_shapes
: *%
shared_nametraining/Adam/m_10_1
y
5training/Adam/m_10_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/m_10_1*
_output_shapes
: 
f
training/Adam/m_10_1/AssignAssignVariableOptraining/Adam/m_10_1training/Adam/m_10*
dtype0
z
(training/Adam/m_10_1/Read/ReadVariableOpReadVariableOptraining/Adam/m_10_1*
dtype0*
_output_shapes	
:�
a
training/Adam/m_11Const*
valueB�*    *
dtype0*
_output_shapes	
:�
�
training/Adam/m_11_1VarHandleOp*
dtype0*
_output_shapes
: *%
shared_nametraining/Adam/m_11_1*'
_class
loc:@training/Adam/m_11_1*
	container *
shape:�
y
5training/Adam/m_11_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/m_11_1*
_output_shapes
: 
f
training/Adam/m_11_1/AssignAssignVariableOptraining/Adam/m_11_1training/Adam/m_11*
dtype0
z
(training/Adam/m_11_1/Read/ReadVariableOpReadVariableOptraining/Adam/m_11_1*
dtype0*
_output_shapes	
:�
s
"training/Adam/m_12/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB"�   �   
]
training/Adam/m_12/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
�
training/Adam/m_12Fill"training/Adam/m_12/shape_as_tensortraining/Adam/m_12/Const* 
_output_shapes
:
��*
T0*

index_type0
�
training/Adam/m_12_1VarHandleOp*%
shared_nametraining/Adam/m_12_1*'
_class
loc:@training/Adam/m_12_1*
	container *
shape:
��*
dtype0*
_output_shapes
: 
y
5training/Adam/m_12_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/m_12_1*
_output_shapes
: 
f
training/Adam/m_12_1/AssignAssignVariableOptraining/Adam/m_12_1training/Adam/m_12*
dtype0

(training/Adam/m_12_1/Read/ReadVariableOpReadVariableOptraining/Adam/m_12_1*
dtype0* 
_output_shapes
:
��
a
training/Adam/m_13Const*
valueB�*    *
dtype0*
_output_shapes	
:�
�
training/Adam/m_13_1VarHandleOp*%
shared_nametraining/Adam/m_13_1*'
_class
loc:@training/Adam/m_13_1*
	container *
shape:�*
dtype0*
_output_shapes
: 
y
5training/Adam/m_13_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/m_13_1*
_output_shapes
: 
f
training/Adam/m_13_1/AssignAssignVariableOptraining/Adam/m_13_1training/Adam/m_13*
dtype0
z
(training/Adam/m_13_1/Read/ReadVariableOpReadVariableOptraining/Adam/m_13_1*
dtype0*
_output_shapes	
:�
a
training/Adam/m_14Const*
valueB�*    *
dtype0*
_output_shapes	
:�
�
training/Adam/m_14_1VarHandleOp*
dtype0*
_output_shapes
: *%
shared_nametraining/Adam/m_14_1*'
_class
loc:@training/Adam/m_14_1*
	container *
shape:�
y
5training/Adam/m_14_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/m_14_1*
_output_shapes
: 
f
training/Adam/m_14_1/AssignAssignVariableOptraining/Adam/m_14_1training/Adam/m_14*
dtype0
z
(training/Adam/m_14_1/Read/ReadVariableOpReadVariableOptraining/Adam/m_14_1*
dtype0*
_output_shapes	
:�
a
training/Adam/m_15Const*
dtype0*
_output_shapes	
:�*
valueB�*    
�
training/Adam/m_15_1VarHandleOp*
	container *
shape:�*
dtype0*
_output_shapes
: *%
shared_nametraining/Adam/m_15_1*'
_class
loc:@training/Adam/m_15_1
y
5training/Adam/m_15_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/m_15_1*
_output_shapes
: 
f
training/Adam/m_15_1/AssignAssignVariableOptraining/Adam/m_15_1training/Adam/m_15*
dtype0
z
(training/Adam/m_15_1/Read/ReadVariableOpReadVariableOptraining/Adam/m_15_1*
dtype0*
_output_shapes	
:�
s
"training/Adam/m_16/shape_as_tensorConst*
valueB"�   +   *
dtype0*
_output_shapes
:
]
training/Adam/m_16/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/m_16Fill"training/Adam/m_16/shape_as_tensortraining/Adam/m_16/Const*
_output_shapes
:	�+*
T0*

index_type0
�
training/Adam/m_16_1VarHandleOp*%
shared_nametraining/Adam/m_16_1*'
_class
loc:@training/Adam/m_16_1*
	container *
shape:	�+*
dtype0*
_output_shapes
: 
y
5training/Adam/m_16_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/m_16_1*
_output_shapes
: 
f
training/Adam/m_16_1/AssignAssignVariableOptraining/Adam/m_16_1training/Adam/m_16*
dtype0
~
(training/Adam/m_16_1/Read/ReadVariableOpReadVariableOptraining/Adam/m_16_1*
dtype0*
_output_shapes
:	�+
_
training/Adam/m_17Const*
dtype0*
_output_shapes
:+*
valueB+*    
�
training/Adam/m_17_1VarHandleOp*%
shared_nametraining/Adam/m_17_1*'
_class
loc:@training/Adam/m_17_1*
	container *
shape:+*
dtype0*
_output_shapes
: 
y
5training/Adam/m_17_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/m_17_1*
_output_shapes
: 
f
training/Adam/m_17_1/AssignAssignVariableOptraining/Adam/m_17_1training/Adam/m_17*
dtype0
y
(training/Adam/m_17_1/Read/ReadVariableOpReadVariableOptraining/Adam/m_17_1*
dtype0*
_output_shapes
:+
r
!training/Adam/v_0/shape_as_tensorConst*
valueB"   �   *
dtype0*
_output_shapes
:
\
training/Adam/v_0/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
�
training/Adam/v_0Fill!training/Adam/v_0/shape_as_tensortraining/Adam/v_0/Const* 
_output_shapes
:
��*
T0*

index_type0
�
training/Adam/v_0_1VarHandleOp*
shape:
��*
dtype0*
_output_shapes
: *$
shared_nametraining/Adam/v_0_1*&
_class
loc:@training/Adam/v_0_1*
	container 
w
4training/Adam/v_0_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/v_0_1*
_output_shapes
: 
c
training/Adam/v_0_1/AssignAssignVariableOptraining/Adam/v_0_1training/Adam/v_0*
dtype0
}
'training/Adam/v_0_1/Read/ReadVariableOpReadVariableOptraining/Adam/v_0_1*
dtype0* 
_output_shapes
:
��
`
training/Adam/v_1Const*
valueB�*    *
dtype0*
_output_shapes	
:�
�
training/Adam/v_1_1VarHandleOp*
shape:�*
dtype0*
_output_shapes
: *$
shared_nametraining/Adam/v_1_1*&
_class
loc:@training/Adam/v_1_1*
	container 
w
4training/Adam/v_1_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/v_1_1*
_output_shapes
: 
c
training/Adam/v_1_1/AssignAssignVariableOptraining/Adam/v_1_1training/Adam/v_1*
dtype0
x
'training/Adam/v_1_1/Read/ReadVariableOpReadVariableOptraining/Adam/v_1_1*
dtype0*
_output_shapes	
:�
`
training/Adam/v_2Const*
dtype0*
_output_shapes	
:�*
valueB�*    
�
training/Adam/v_2_1VarHandleOp*
shape:�*
dtype0*
_output_shapes
: *$
shared_nametraining/Adam/v_2_1*&
_class
loc:@training/Adam/v_2_1*
	container 
w
4training/Adam/v_2_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/v_2_1*
_output_shapes
: 
c
training/Adam/v_2_1/AssignAssignVariableOptraining/Adam/v_2_1training/Adam/v_2*
dtype0
x
'training/Adam/v_2_1/Read/ReadVariableOpReadVariableOptraining/Adam/v_2_1*
dtype0*
_output_shapes	
:�
`
training/Adam/v_3Const*
valueB�*    *
dtype0*
_output_shapes	
:�
�
training/Adam/v_3_1VarHandleOp*&
_class
loc:@training/Adam/v_3_1*
	container *
shape:�*
dtype0*
_output_shapes
: *$
shared_nametraining/Adam/v_3_1
w
4training/Adam/v_3_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/v_3_1*
_output_shapes
: 
c
training/Adam/v_3_1/AssignAssignVariableOptraining/Adam/v_3_1training/Adam/v_3*
dtype0
x
'training/Adam/v_3_1/Read/ReadVariableOpReadVariableOptraining/Adam/v_3_1*
dtype0*
_output_shapes	
:�
r
!training/Adam/v_4/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB"�   �   
\
training/Adam/v_4/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/v_4Fill!training/Adam/v_4/shape_as_tensortraining/Adam/v_4/Const*
T0*

index_type0* 
_output_shapes
:
��
�
training/Adam/v_4_1VarHandleOp*
	container *
shape:
��*
dtype0*
_output_shapes
: *$
shared_nametraining/Adam/v_4_1*&
_class
loc:@training/Adam/v_4_1
w
4training/Adam/v_4_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/v_4_1*
_output_shapes
: 
c
training/Adam/v_4_1/AssignAssignVariableOptraining/Adam/v_4_1training/Adam/v_4*
dtype0
}
'training/Adam/v_4_1/Read/ReadVariableOpReadVariableOptraining/Adam/v_4_1*
dtype0* 
_output_shapes
:
��
`
training/Adam/v_5Const*
dtype0*
_output_shapes	
:�*
valueB�*    
�
training/Adam/v_5_1VarHandleOp*$
shared_nametraining/Adam/v_5_1*&
_class
loc:@training/Adam/v_5_1*
	container *
shape:�*
dtype0*
_output_shapes
: 
w
4training/Adam/v_5_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/v_5_1*
_output_shapes
: 
c
training/Adam/v_5_1/AssignAssignVariableOptraining/Adam/v_5_1training/Adam/v_5*
dtype0
x
'training/Adam/v_5_1/Read/ReadVariableOpReadVariableOptraining/Adam/v_5_1*
dtype0*
_output_shapes	
:�
`
training/Adam/v_6Const*
valueB�*    *
dtype0*
_output_shapes	
:�
�
training/Adam/v_6_1VarHandleOp*$
shared_nametraining/Adam/v_6_1*&
_class
loc:@training/Adam/v_6_1*
	container *
shape:�*
dtype0*
_output_shapes
: 
w
4training/Adam/v_6_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/v_6_1*
_output_shapes
: 
c
training/Adam/v_6_1/AssignAssignVariableOptraining/Adam/v_6_1training/Adam/v_6*
dtype0
x
'training/Adam/v_6_1/Read/ReadVariableOpReadVariableOptraining/Adam/v_6_1*
dtype0*
_output_shapes	
:�
`
training/Adam/v_7Const*
valueB�*    *
dtype0*
_output_shapes	
:�
�
training/Adam/v_7_1VarHandleOp*
shape:�*
dtype0*
_output_shapes
: *$
shared_nametraining/Adam/v_7_1*&
_class
loc:@training/Adam/v_7_1*
	container 
w
4training/Adam/v_7_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/v_7_1*
_output_shapes
: 
c
training/Adam/v_7_1/AssignAssignVariableOptraining/Adam/v_7_1training/Adam/v_7*
dtype0
x
'training/Adam/v_7_1/Read/ReadVariableOpReadVariableOptraining/Adam/v_7_1*
dtype0*
_output_shapes	
:�
r
!training/Adam/v_8/shape_as_tensorConst*
valueB"�   �   *
dtype0*
_output_shapes
:
\
training/Adam/v_8/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/v_8Fill!training/Adam/v_8/shape_as_tensortraining/Adam/v_8/Const*
T0*

index_type0* 
_output_shapes
:
��
�
training/Adam/v_8_1VarHandleOp*&
_class
loc:@training/Adam/v_8_1*
	container *
shape:
��*
dtype0*
_output_shapes
: *$
shared_nametraining/Adam/v_8_1
w
4training/Adam/v_8_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/v_8_1*
_output_shapes
: 
c
training/Adam/v_8_1/AssignAssignVariableOptraining/Adam/v_8_1training/Adam/v_8*
dtype0
}
'training/Adam/v_8_1/Read/ReadVariableOpReadVariableOptraining/Adam/v_8_1*
dtype0* 
_output_shapes
:
��
`
training/Adam/v_9Const*
valueB�*    *
dtype0*
_output_shapes	
:�
�
training/Adam/v_9_1VarHandleOp*
dtype0*
_output_shapes
: *$
shared_nametraining/Adam/v_9_1*&
_class
loc:@training/Adam/v_9_1*
	container *
shape:�
w
4training/Adam/v_9_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/v_9_1*
_output_shapes
: 
c
training/Adam/v_9_1/AssignAssignVariableOptraining/Adam/v_9_1training/Adam/v_9*
dtype0
x
'training/Adam/v_9_1/Read/ReadVariableOpReadVariableOptraining/Adam/v_9_1*
dtype0*
_output_shapes	
:�
a
training/Adam/v_10Const*
valueB�*    *
dtype0*
_output_shapes	
:�
�
training/Adam/v_10_1VarHandleOp*
dtype0*
_output_shapes
: *%
shared_nametraining/Adam/v_10_1*'
_class
loc:@training/Adam/v_10_1*
	container *
shape:�
y
5training/Adam/v_10_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/v_10_1*
_output_shapes
: 
f
training/Adam/v_10_1/AssignAssignVariableOptraining/Adam/v_10_1training/Adam/v_10*
dtype0
z
(training/Adam/v_10_1/Read/ReadVariableOpReadVariableOptraining/Adam/v_10_1*
dtype0*
_output_shapes	
:�
a
training/Adam/v_11Const*
valueB�*    *
dtype0*
_output_shapes	
:�
�
training/Adam/v_11_1VarHandleOp*
shape:�*
dtype0*
_output_shapes
: *%
shared_nametraining/Adam/v_11_1*'
_class
loc:@training/Adam/v_11_1*
	container 
y
5training/Adam/v_11_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/v_11_1*
_output_shapes
: 
f
training/Adam/v_11_1/AssignAssignVariableOptraining/Adam/v_11_1training/Adam/v_11*
dtype0
z
(training/Adam/v_11_1/Read/ReadVariableOpReadVariableOptraining/Adam/v_11_1*
dtype0*
_output_shapes	
:�
s
"training/Adam/v_12/shape_as_tensorConst*
valueB"�   �   *
dtype0*
_output_shapes
:
]
training/Adam/v_12/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
�
training/Adam/v_12Fill"training/Adam/v_12/shape_as_tensortraining/Adam/v_12/Const* 
_output_shapes
:
��*
T0*

index_type0
�
training/Adam/v_12_1VarHandleOp*
dtype0*
_output_shapes
: *%
shared_nametraining/Adam/v_12_1*'
_class
loc:@training/Adam/v_12_1*
	container *
shape:
��
y
5training/Adam/v_12_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/v_12_1*
_output_shapes
: 
f
training/Adam/v_12_1/AssignAssignVariableOptraining/Adam/v_12_1training/Adam/v_12*
dtype0

(training/Adam/v_12_1/Read/ReadVariableOpReadVariableOptraining/Adam/v_12_1*
dtype0* 
_output_shapes
:
��
a
training/Adam/v_13Const*
valueB�*    *
dtype0*
_output_shapes	
:�
�
training/Adam/v_13_1VarHandleOp*
shape:�*
dtype0*
_output_shapes
: *%
shared_nametraining/Adam/v_13_1*'
_class
loc:@training/Adam/v_13_1*
	container 
y
5training/Adam/v_13_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/v_13_1*
_output_shapes
: 
f
training/Adam/v_13_1/AssignAssignVariableOptraining/Adam/v_13_1training/Adam/v_13*
dtype0
z
(training/Adam/v_13_1/Read/ReadVariableOpReadVariableOptraining/Adam/v_13_1*
dtype0*
_output_shapes	
:�
a
training/Adam/v_14Const*
valueB�*    *
dtype0*
_output_shapes	
:�
�
training/Adam/v_14_1VarHandleOp*'
_class
loc:@training/Adam/v_14_1*
	container *
shape:�*
dtype0*
_output_shapes
: *%
shared_nametraining/Adam/v_14_1
y
5training/Adam/v_14_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/v_14_1*
_output_shapes
: 
f
training/Adam/v_14_1/AssignAssignVariableOptraining/Adam/v_14_1training/Adam/v_14*
dtype0
z
(training/Adam/v_14_1/Read/ReadVariableOpReadVariableOptraining/Adam/v_14_1*
dtype0*
_output_shapes	
:�
a
training/Adam/v_15Const*
valueB�*    *
dtype0*
_output_shapes	
:�
�
training/Adam/v_15_1VarHandleOp*
shape:�*
dtype0*
_output_shapes
: *%
shared_nametraining/Adam/v_15_1*'
_class
loc:@training/Adam/v_15_1*
	container 
y
5training/Adam/v_15_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/v_15_1*
_output_shapes
: 
f
training/Adam/v_15_1/AssignAssignVariableOptraining/Adam/v_15_1training/Adam/v_15*
dtype0
z
(training/Adam/v_15_1/Read/ReadVariableOpReadVariableOptraining/Adam/v_15_1*
dtype0*
_output_shapes	
:�
s
"training/Adam/v_16/shape_as_tensorConst*
valueB"�   +   *
dtype0*
_output_shapes
:
]
training/Adam/v_16/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/v_16Fill"training/Adam/v_16/shape_as_tensortraining/Adam/v_16/Const*
_output_shapes
:	�+*
T0*

index_type0
�
training/Adam/v_16_1VarHandleOp*
dtype0*
_output_shapes
: *%
shared_nametraining/Adam/v_16_1*'
_class
loc:@training/Adam/v_16_1*
	container *
shape:	�+
y
5training/Adam/v_16_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/v_16_1*
_output_shapes
: 
f
training/Adam/v_16_1/AssignAssignVariableOptraining/Adam/v_16_1training/Adam/v_16*
dtype0
~
(training/Adam/v_16_1/Read/ReadVariableOpReadVariableOptraining/Adam/v_16_1*
dtype0*
_output_shapes
:	�+
_
training/Adam/v_17Const*
valueB+*    *
dtype0*
_output_shapes
:+
�
training/Adam/v_17_1VarHandleOp*
dtype0*
_output_shapes
: *%
shared_nametraining/Adam/v_17_1*'
_class
loc:@training/Adam/v_17_1*
	container *
shape:+
y
5training/Adam/v_17_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/v_17_1*
_output_shapes
: 
f
training/Adam/v_17_1/AssignAssignVariableOptraining/Adam/v_17_1training/Adam/v_17*
dtype0
y
(training/Adam/v_17_1/Read/ReadVariableOpReadVariableOptraining/Adam/v_17_1*
dtype0*
_output_shapes
:+
n
$training/Adam/vhat_0/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
_
training/Adam/vhat_0/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/vhat_0Fill$training/Adam/vhat_0/shape_as_tensortraining/Adam/vhat_0/Const*
T0*

index_type0*
_output_shapes
:
�
training/Adam/vhat_0_1VarHandleOp*'
shared_nametraining/Adam/vhat_0_1*)
_class
loc:@training/Adam/vhat_0_1*
	container *
shape:*
dtype0*
_output_shapes
: 
}
7training/Adam/vhat_0_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/vhat_0_1*
_output_shapes
: 
l
training/Adam/vhat_0_1/AssignAssignVariableOptraining/Adam/vhat_0_1training/Adam/vhat_0*
dtype0
}
*training/Adam/vhat_0_1/Read/ReadVariableOpReadVariableOptraining/Adam/vhat_0_1*
dtype0*
_output_shapes
:
n
$training/Adam/vhat_1/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
_
training/Adam/vhat_1/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
�
training/Adam/vhat_1Fill$training/Adam/vhat_1/shape_as_tensortraining/Adam/vhat_1/Const*
T0*

index_type0*
_output_shapes
:
�
training/Adam/vhat_1_1VarHandleOp*'
shared_nametraining/Adam/vhat_1_1*)
_class
loc:@training/Adam/vhat_1_1*
	container *
shape:*
dtype0*
_output_shapes
: 
}
7training/Adam/vhat_1_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/vhat_1_1*
_output_shapes
: 
l
training/Adam/vhat_1_1/AssignAssignVariableOptraining/Adam/vhat_1_1training/Adam/vhat_1*
dtype0
}
*training/Adam/vhat_1_1/Read/ReadVariableOpReadVariableOptraining/Adam/vhat_1_1*
dtype0*
_output_shapes
:
n
$training/Adam/vhat_2/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
_
training/Adam/vhat_2/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/vhat_2Fill$training/Adam/vhat_2/shape_as_tensortraining/Adam/vhat_2/Const*
T0*

index_type0*
_output_shapes
:
�
training/Adam/vhat_2_1VarHandleOp*
dtype0*
_output_shapes
: *'
shared_nametraining/Adam/vhat_2_1*)
_class
loc:@training/Adam/vhat_2_1*
	container *
shape:
}
7training/Adam/vhat_2_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/vhat_2_1*
_output_shapes
: 
l
training/Adam/vhat_2_1/AssignAssignVariableOptraining/Adam/vhat_2_1training/Adam/vhat_2*
dtype0
}
*training/Adam/vhat_2_1/Read/ReadVariableOpReadVariableOptraining/Adam/vhat_2_1*
dtype0*
_output_shapes
:
n
$training/Adam/vhat_3/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
_
training/Adam/vhat_3/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/vhat_3Fill$training/Adam/vhat_3/shape_as_tensortraining/Adam/vhat_3/Const*
_output_shapes
:*
T0*

index_type0
�
training/Adam/vhat_3_1VarHandleOp*
dtype0*
_output_shapes
: *'
shared_nametraining/Adam/vhat_3_1*)
_class
loc:@training/Adam/vhat_3_1*
	container *
shape:
}
7training/Adam/vhat_3_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/vhat_3_1*
_output_shapes
: 
l
training/Adam/vhat_3_1/AssignAssignVariableOptraining/Adam/vhat_3_1training/Adam/vhat_3*
dtype0
}
*training/Adam/vhat_3_1/Read/ReadVariableOpReadVariableOptraining/Adam/vhat_3_1*
dtype0*
_output_shapes
:
n
$training/Adam/vhat_4/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
_
training/Adam/vhat_4/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/vhat_4Fill$training/Adam/vhat_4/shape_as_tensortraining/Adam/vhat_4/Const*
T0*

index_type0*
_output_shapes
:
�
training/Adam/vhat_4_1VarHandleOp*
dtype0*
_output_shapes
: *'
shared_nametraining/Adam/vhat_4_1*)
_class
loc:@training/Adam/vhat_4_1*
	container *
shape:
}
7training/Adam/vhat_4_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/vhat_4_1*
_output_shapes
: 
l
training/Adam/vhat_4_1/AssignAssignVariableOptraining/Adam/vhat_4_1training/Adam/vhat_4*
dtype0
}
*training/Adam/vhat_4_1/Read/ReadVariableOpReadVariableOptraining/Adam/vhat_4_1*
dtype0*
_output_shapes
:
n
$training/Adam/vhat_5/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
_
training/Adam/vhat_5/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
�
training/Adam/vhat_5Fill$training/Adam/vhat_5/shape_as_tensortraining/Adam/vhat_5/Const*
T0*

index_type0*
_output_shapes
:
�
training/Adam/vhat_5_1VarHandleOp*
	container *
shape:*
dtype0*
_output_shapes
: *'
shared_nametraining/Adam/vhat_5_1*)
_class
loc:@training/Adam/vhat_5_1
}
7training/Adam/vhat_5_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/vhat_5_1*
_output_shapes
: 
l
training/Adam/vhat_5_1/AssignAssignVariableOptraining/Adam/vhat_5_1training/Adam/vhat_5*
dtype0
}
*training/Adam/vhat_5_1/Read/ReadVariableOpReadVariableOptraining/Adam/vhat_5_1*
dtype0*
_output_shapes
:
n
$training/Adam/vhat_6/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
_
training/Adam/vhat_6/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/vhat_6Fill$training/Adam/vhat_6/shape_as_tensortraining/Adam/vhat_6/Const*
T0*

index_type0*
_output_shapes
:
�
training/Adam/vhat_6_1VarHandleOp*
dtype0*
_output_shapes
: *'
shared_nametraining/Adam/vhat_6_1*)
_class
loc:@training/Adam/vhat_6_1*
	container *
shape:
}
7training/Adam/vhat_6_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/vhat_6_1*
_output_shapes
: 
l
training/Adam/vhat_6_1/AssignAssignVariableOptraining/Adam/vhat_6_1training/Adam/vhat_6*
dtype0
}
*training/Adam/vhat_6_1/Read/ReadVariableOpReadVariableOptraining/Adam/vhat_6_1*
dtype0*
_output_shapes
:
n
$training/Adam/vhat_7/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
_
training/Adam/vhat_7/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/vhat_7Fill$training/Adam/vhat_7/shape_as_tensortraining/Adam/vhat_7/Const*
T0*

index_type0*
_output_shapes
:
�
training/Adam/vhat_7_1VarHandleOp*
dtype0*
_output_shapes
: *'
shared_nametraining/Adam/vhat_7_1*)
_class
loc:@training/Adam/vhat_7_1*
	container *
shape:
}
7training/Adam/vhat_7_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/vhat_7_1*
_output_shapes
: 
l
training/Adam/vhat_7_1/AssignAssignVariableOptraining/Adam/vhat_7_1training/Adam/vhat_7*
dtype0
}
*training/Adam/vhat_7_1/Read/ReadVariableOpReadVariableOptraining/Adam/vhat_7_1*
dtype0*
_output_shapes
:
n
$training/Adam/vhat_8/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
_
training/Adam/vhat_8/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/vhat_8Fill$training/Adam/vhat_8/shape_as_tensortraining/Adam/vhat_8/Const*
T0*

index_type0*
_output_shapes
:
�
training/Adam/vhat_8_1VarHandleOp*
shape:*
dtype0*
_output_shapes
: *'
shared_nametraining/Adam/vhat_8_1*)
_class
loc:@training/Adam/vhat_8_1*
	container 
}
7training/Adam/vhat_8_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/vhat_8_1*
_output_shapes
: 
l
training/Adam/vhat_8_1/AssignAssignVariableOptraining/Adam/vhat_8_1training/Adam/vhat_8*
dtype0
}
*training/Adam/vhat_8_1/Read/ReadVariableOpReadVariableOptraining/Adam/vhat_8_1*
dtype0*
_output_shapes
:
n
$training/Adam/vhat_9/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
_
training/Adam/vhat_9/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/vhat_9Fill$training/Adam/vhat_9/shape_as_tensortraining/Adam/vhat_9/Const*
T0*

index_type0*
_output_shapes
:
�
training/Adam/vhat_9_1VarHandleOp*'
shared_nametraining/Adam/vhat_9_1*)
_class
loc:@training/Adam/vhat_9_1*
	container *
shape:*
dtype0*
_output_shapes
: 
}
7training/Adam/vhat_9_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/vhat_9_1*
_output_shapes
: 
l
training/Adam/vhat_9_1/AssignAssignVariableOptraining/Adam/vhat_9_1training/Adam/vhat_9*
dtype0
}
*training/Adam/vhat_9_1/Read/ReadVariableOpReadVariableOptraining/Adam/vhat_9_1*
dtype0*
_output_shapes
:
o
%training/Adam/vhat_10/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
`
training/Adam/vhat_10/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/vhat_10Fill%training/Adam/vhat_10/shape_as_tensortraining/Adam/vhat_10/Const*
_output_shapes
:*
T0*

index_type0
�
training/Adam/vhat_10_1VarHandleOp*(
shared_nametraining/Adam/vhat_10_1**
_class 
loc:@training/Adam/vhat_10_1*
	container *
shape:*
dtype0*
_output_shapes
: 

8training/Adam/vhat_10_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/vhat_10_1*
_output_shapes
: 
o
training/Adam/vhat_10_1/AssignAssignVariableOptraining/Adam/vhat_10_1training/Adam/vhat_10*
dtype0

+training/Adam/vhat_10_1/Read/ReadVariableOpReadVariableOptraining/Adam/vhat_10_1*
dtype0*
_output_shapes
:
o
%training/Adam/vhat_11/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
`
training/Adam/vhat_11/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/vhat_11Fill%training/Adam/vhat_11/shape_as_tensortraining/Adam/vhat_11/Const*
T0*

index_type0*
_output_shapes
:
�
training/Adam/vhat_11_1VarHandleOp*
	container *
shape:*
dtype0*
_output_shapes
: *(
shared_nametraining/Adam/vhat_11_1**
_class 
loc:@training/Adam/vhat_11_1

8training/Adam/vhat_11_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/vhat_11_1*
_output_shapes
: 
o
training/Adam/vhat_11_1/AssignAssignVariableOptraining/Adam/vhat_11_1training/Adam/vhat_11*
dtype0

+training/Adam/vhat_11_1/Read/ReadVariableOpReadVariableOptraining/Adam/vhat_11_1*
dtype0*
_output_shapes
:
o
%training/Adam/vhat_12/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
`
training/Adam/vhat_12/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/vhat_12Fill%training/Adam/vhat_12/shape_as_tensortraining/Adam/vhat_12/Const*
T0*

index_type0*
_output_shapes
:
�
training/Adam/vhat_12_1VarHandleOp*(
shared_nametraining/Adam/vhat_12_1**
_class 
loc:@training/Adam/vhat_12_1*
	container *
shape:*
dtype0*
_output_shapes
: 

8training/Adam/vhat_12_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/vhat_12_1*
_output_shapes
: 
o
training/Adam/vhat_12_1/AssignAssignVariableOptraining/Adam/vhat_12_1training/Adam/vhat_12*
dtype0

+training/Adam/vhat_12_1/Read/ReadVariableOpReadVariableOptraining/Adam/vhat_12_1*
dtype0*
_output_shapes
:
o
%training/Adam/vhat_13/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
`
training/Adam/vhat_13/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/vhat_13Fill%training/Adam/vhat_13/shape_as_tensortraining/Adam/vhat_13/Const*
T0*

index_type0*
_output_shapes
:
�
training/Adam/vhat_13_1VarHandleOp**
_class 
loc:@training/Adam/vhat_13_1*
	container *
shape:*
dtype0*
_output_shapes
: *(
shared_nametraining/Adam/vhat_13_1

8training/Adam/vhat_13_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/vhat_13_1*
_output_shapes
: 
o
training/Adam/vhat_13_1/AssignAssignVariableOptraining/Adam/vhat_13_1training/Adam/vhat_13*
dtype0

+training/Adam/vhat_13_1/Read/ReadVariableOpReadVariableOptraining/Adam/vhat_13_1*
dtype0*
_output_shapes
:
o
%training/Adam/vhat_14/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
`
training/Adam/vhat_14/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/vhat_14Fill%training/Adam/vhat_14/shape_as_tensortraining/Adam/vhat_14/Const*
T0*

index_type0*
_output_shapes
:
�
training/Adam/vhat_14_1VarHandleOp*
dtype0*
_output_shapes
: *(
shared_nametraining/Adam/vhat_14_1**
_class 
loc:@training/Adam/vhat_14_1*
	container *
shape:

8training/Adam/vhat_14_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/vhat_14_1*
_output_shapes
: 
o
training/Adam/vhat_14_1/AssignAssignVariableOptraining/Adam/vhat_14_1training/Adam/vhat_14*
dtype0

+training/Adam/vhat_14_1/Read/ReadVariableOpReadVariableOptraining/Adam/vhat_14_1*
dtype0*
_output_shapes
:
o
%training/Adam/vhat_15/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
`
training/Adam/vhat_15/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/vhat_15Fill%training/Adam/vhat_15/shape_as_tensortraining/Adam/vhat_15/Const*
T0*

index_type0*
_output_shapes
:
�
training/Adam/vhat_15_1VarHandleOp*
shape:*
dtype0*
_output_shapes
: *(
shared_nametraining/Adam/vhat_15_1**
_class 
loc:@training/Adam/vhat_15_1*
	container 

8training/Adam/vhat_15_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/vhat_15_1*
_output_shapes
: 
o
training/Adam/vhat_15_1/AssignAssignVariableOptraining/Adam/vhat_15_1training/Adam/vhat_15*
dtype0

+training/Adam/vhat_15_1/Read/ReadVariableOpReadVariableOptraining/Adam/vhat_15_1*
dtype0*
_output_shapes
:
o
%training/Adam/vhat_16/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
`
training/Adam/vhat_16/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/vhat_16Fill%training/Adam/vhat_16/shape_as_tensortraining/Adam/vhat_16/Const*
T0*

index_type0*
_output_shapes
:
�
training/Adam/vhat_16_1VarHandleOp*
shape:*
dtype0*
_output_shapes
: *(
shared_nametraining/Adam/vhat_16_1**
_class 
loc:@training/Adam/vhat_16_1*
	container 

8training/Adam/vhat_16_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/vhat_16_1*
_output_shapes
: 
o
training/Adam/vhat_16_1/AssignAssignVariableOptraining/Adam/vhat_16_1training/Adam/vhat_16*
dtype0

+training/Adam/vhat_16_1/Read/ReadVariableOpReadVariableOptraining/Adam/vhat_16_1*
dtype0*
_output_shapes
:
o
%training/Adam/vhat_17/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
`
training/Adam/vhat_17/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
�
training/Adam/vhat_17Fill%training/Adam/vhat_17/shape_as_tensortraining/Adam/vhat_17/Const*
T0*

index_type0*
_output_shapes
:
�
training/Adam/vhat_17_1VarHandleOp*(
shared_nametraining/Adam/vhat_17_1**
_class 
loc:@training/Adam/vhat_17_1*
	container *
shape:*
dtype0*
_output_shapes
: 

8training/Adam/vhat_17_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/vhat_17_1*
_output_shapes
: 
o
training/Adam/vhat_17_1/AssignAssignVariableOptraining/Adam/vhat_17_1training/Adam/vhat_17*
dtype0

+training/Adam/vhat_17_1/Read/ReadVariableOpReadVariableOptraining/Adam/vhat_17_1*
dtype0*
_output_shapes
:
b
training/Adam/ReadVariableOp_2ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
x
"training/Adam/mul_1/ReadVariableOpReadVariableOptraining/Adam/m_0_1*
dtype0* 
_output_shapes
:
��
�
training/Adam/mul_1Multraining/Adam/ReadVariableOp_2"training/Adam/mul_1/ReadVariableOp*
T0* 
_output_shapes
:
��
b
training/Adam/ReadVariableOp_3ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
Z
training/Adam/sub_2/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
r
training/Adam/sub_2Subtraining/Adam/sub_2/xtraining/Adam/ReadVariableOp_3*
_output_shapes
: *
T0
�
training/Adam/mul_2Multraining/Adam/sub_24training/Adam/gradients/dense_1/MatMul_grad/MatMul_1* 
_output_shapes
:
��*
T0
q
training/Adam/add_1AddV2training/Adam/mul_1training/Adam/mul_2*
T0* 
_output_shapes
:
��
b
training/Adam/ReadVariableOp_4ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
x
"training/Adam/mul_3/ReadVariableOpReadVariableOptraining/Adam/v_0_1*
dtype0* 
_output_shapes
:
��
�
training/Adam/mul_3Multraining/Adam/ReadVariableOp_4"training/Adam/mul_3/ReadVariableOp* 
_output_shapes
:
��*
T0
b
training/Adam/ReadVariableOp_5ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
Z
training/Adam/sub_3/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
r
training/Adam/sub_3Subtraining/Adam/sub_3/xtraining/Adam/ReadVariableOp_5*
T0*
_output_shapes
: 

training/Adam/SquareSquare4training/Adam/gradients/dense_1/MatMul_grad/MatMul_1* 
_output_shapes
:
��*
T0
p
training/Adam/mul_4Multraining/Adam/sub_3training/Adam/Square* 
_output_shapes
:
��*
T0
q
training/Adam/add_2AddV2training/Adam/mul_3training/Adam/mul_4* 
_output_shapes
:
��*
T0
m
training/Adam/mul_5Multraining/Adam/multraining/Adam/add_1*
T0* 
_output_shapes
:
��
Z
training/Adam/Const_3Const*
dtype0*
_output_shapes
: *
valueB
 *    
Z
training/Adam/Const_4Const*
valueB
 *  �*
dtype0*
_output_shapes
: 
�
%training/Adam/clip_by_value_1/MinimumMinimumtraining/Adam/add_2training/Adam/Const_4* 
_output_shapes
:
��*
T0
�
training/Adam/clip_by_value_1Maximum%training/Adam/clip_by_value_1/Minimumtraining/Adam/Const_3* 
_output_shapes
:
��*
T0
f
training/Adam/Sqrt_1Sqrttraining/Adam/clip_by_value_1*
T0* 
_output_shapes
:
��
Z
training/Adam/add_3/yConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
t
training/Adam/add_3AddV2training/Adam/Sqrt_1training/Adam/add_3/y*
T0* 
_output_shapes
:
��
w
training/Adam/truediv_1RealDivtraining/Adam/mul_5training/Adam/add_3* 
_output_shapes
:
��*
T0
o
training/Adam/ReadVariableOp_6ReadVariableOpdense_1/kernel*
dtype0* 
_output_shapes
:
��
~
training/Adam/sub_4Subtraining/Adam/ReadVariableOp_6training/Adam/truediv_1*
T0* 
_output_shapes
:
��
i
training/Adam/AssignVariableOpAssignVariableOptraining/Adam/m_0_1training/Adam/add_1*
dtype0
�
training/Adam/ReadVariableOp_7ReadVariableOptraining/Adam/m_0_1^training/Adam/AssignVariableOp*
dtype0* 
_output_shapes
:
��
k
 training/Adam/AssignVariableOp_1AssignVariableOptraining/Adam/v_0_1training/Adam/add_2*
dtype0
�
training/Adam/ReadVariableOp_8ReadVariableOptraining/Adam/v_0_1!^training/Adam/AssignVariableOp_1*
dtype0* 
_output_shapes
:
��
f
 training/Adam/AssignVariableOp_2AssignVariableOpdense_1/kerneltraining/Adam/sub_4*
dtype0
�
training/Adam/ReadVariableOp_9ReadVariableOpdense_1/kernel!^training/Adam/AssignVariableOp_2*
dtype0* 
_output_shapes
:
��
c
training/Adam/ReadVariableOp_10ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
s
"training/Adam/mul_6/ReadVariableOpReadVariableOptraining/Adam/m_1_1*
dtype0*
_output_shapes	
:�
�
training/Adam/mul_6Multraining/Adam/ReadVariableOp_10"training/Adam/mul_6/ReadVariableOp*
T0*
_output_shapes	
:�
c
training/Adam/ReadVariableOp_11ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
Z
training/Adam/sub_5/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
s
training/Adam/sub_5Subtraining/Adam/sub_5/xtraining/Adam/ReadVariableOp_11*
T0*
_output_shapes
: 
�
training/Adam/mul_7Multraining/Adam/sub_58training/Adam/gradients/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�*
T0
l
training/Adam/add_4AddV2training/Adam/mul_6training/Adam/mul_7*
_output_shapes	
:�*
T0
c
training/Adam/ReadVariableOp_12ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
s
"training/Adam/mul_8/ReadVariableOpReadVariableOptraining/Adam/v_1_1*
dtype0*
_output_shapes	
:�
�
training/Adam/mul_8Multraining/Adam/ReadVariableOp_12"training/Adam/mul_8/ReadVariableOp*
T0*
_output_shapes	
:�
c
training/Adam/ReadVariableOp_13ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
Z
training/Adam/sub_6/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
s
training/Adam/sub_6Subtraining/Adam/sub_6/xtraining/Adam/ReadVariableOp_13*
T0*
_output_shapes
: 
�
training/Adam/Square_1Square8training/Adam/gradients/dense_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:�
m
training/Adam/mul_9Multraining/Adam/sub_6training/Adam/Square_1*
_output_shapes	
:�*
T0
l
training/Adam/add_5AddV2training/Adam/mul_8training/Adam/mul_9*
T0*
_output_shapes	
:�
i
training/Adam/mul_10Multraining/Adam/multraining/Adam/add_4*
T0*
_output_shapes	
:�
Z
training/Adam/Const_5Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_6Const*
valueB
 *  �*
dtype0*
_output_shapes
: 
�
%training/Adam/clip_by_value_2/MinimumMinimumtraining/Adam/add_5training/Adam/Const_6*
T0*
_output_shapes	
:�
�
training/Adam/clip_by_value_2Maximum%training/Adam/clip_by_value_2/Minimumtraining/Adam/Const_5*
T0*
_output_shapes	
:�
a
training/Adam/Sqrt_2Sqrttraining/Adam/clip_by_value_2*
T0*
_output_shapes	
:�
Z
training/Adam/add_6/yConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
o
training/Adam/add_6AddV2training/Adam/Sqrt_2training/Adam/add_6/y*
T0*
_output_shapes	
:�
s
training/Adam/truediv_2RealDivtraining/Adam/mul_10training/Adam/add_6*
_output_shapes	
:�*
T0
i
training/Adam/ReadVariableOp_14ReadVariableOpdense_1/bias*
dtype0*
_output_shapes	
:�
z
training/Adam/sub_7Subtraining/Adam/ReadVariableOp_14training/Adam/truediv_2*
T0*
_output_shapes	
:�
k
 training/Adam/AssignVariableOp_3AssignVariableOptraining/Adam/m_1_1training/Adam/add_4*
dtype0
�
training/Adam/ReadVariableOp_15ReadVariableOptraining/Adam/m_1_1!^training/Adam/AssignVariableOp_3*
dtype0*
_output_shapes	
:�
k
 training/Adam/AssignVariableOp_4AssignVariableOptraining/Adam/v_1_1training/Adam/add_5*
dtype0
�
training/Adam/ReadVariableOp_16ReadVariableOptraining/Adam/v_1_1!^training/Adam/AssignVariableOp_4*
dtype0*
_output_shapes	
:�
d
 training/Adam/AssignVariableOp_5AssignVariableOpdense_1/biastraining/Adam/sub_7*
dtype0
�
training/Adam/ReadVariableOp_17ReadVariableOpdense_1/bias!^training/Adam/AssignVariableOp_5*
dtype0*
_output_shapes	
:�
c
training/Adam/ReadVariableOp_18ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
t
#training/Adam/mul_11/ReadVariableOpReadVariableOptraining/Adam/m_2_1*
dtype0*
_output_shapes	
:�
�
training/Adam/mul_11Multraining/Adam/ReadVariableOp_18#training/Adam/mul_11/ReadVariableOp*
T0*
_output_shapes	
:�
c
training/Adam/ReadVariableOp_19ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
Z
training/Adam/sub_8/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
s
training/Adam/sub_8Subtraining/Adam/sub_8/xtraining/Adam/ReadVariableOp_19*
T0*
_output_shapes
: 
w
training/Adam/mul_12Multraining/Adam/sub_8training/Adam/gradients/AddN_20*
T0*
_output_shapes	
:�
n
training/Adam/add_7AddV2training/Adam/mul_11training/Adam/mul_12*
_output_shapes	
:�*
T0
c
training/Adam/ReadVariableOp_20ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
t
#training/Adam/mul_13/ReadVariableOpReadVariableOptraining/Adam/v_2_1*
dtype0*
_output_shapes	
:�
�
training/Adam/mul_13Multraining/Adam/ReadVariableOp_20#training/Adam/mul_13/ReadVariableOp*
T0*
_output_shapes	
:�
c
training/Adam/ReadVariableOp_21ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
Z
training/Adam/sub_9/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
s
training/Adam/sub_9Subtraining/Adam/sub_9/xtraining/Adam/ReadVariableOp_21*
T0*
_output_shapes
: 
g
training/Adam/Square_2Squaretraining/Adam/gradients/AddN_20*
T0*
_output_shapes	
:�
n
training/Adam/mul_14Multraining/Adam/sub_9training/Adam/Square_2*
T0*
_output_shapes	
:�
n
training/Adam/add_8AddV2training/Adam/mul_13training/Adam/mul_14*
_output_shapes	
:�*
T0
i
training/Adam/mul_15Multraining/Adam/multraining/Adam/add_7*
T0*
_output_shapes	
:�
Z
training/Adam/Const_7Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_8Const*
valueB
 *  �*
dtype0*
_output_shapes
: 
�
%training/Adam/clip_by_value_3/MinimumMinimumtraining/Adam/add_8training/Adam/Const_8*
T0*
_output_shapes	
:�
�
training/Adam/clip_by_value_3Maximum%training/Adam/clip_by_value_3/Minimumtraining/Adam/Const_7*
T0*
_output_shapes	
:�
a
training/Adam/Sqrt_3Sqrttraining/Adam/clip_by_value_3*
T0*
_output_shapes	
:�
Z
training/Adam/add_9/yConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
o
training/Adam/add_9AddV2training/Adam/Sqrt_3training/Adam/add_9/y*
T0*
_output_shapes	
:�
s
training/Adam/truediv_3RealDivtraining/Adam/mul_15training/Adam/add_9*
T0*
_output_shapes	
:�
x
training/Adam/ReadVariableOp_22ReadVariableOpbatch_normalization_1/gamma*
dtype0*
_output_shapes	
:�
{
training/Adam/sub_10Subtraining/Adam/ReadVariableOp_22training/Adam/truediv_3*
T0*
_output_shapes	
:�
k
 training/Adam/AssignVariableOp_6AssignVariableOptraining/Adam/m_2_1training/Adam/add_7*
dtype0
�
training/Adam/ReadVariableOp_23ReadVariableOptraining/Adam/m_2_1!^training/Adam/AssignVariableOp_6*
dtype0*
_output_shapes	
:�
k
 training/Adam/AssignVariableOp_7AssignVariableOptraining/Adam/v_2_1training/Adam/add_8*
dtype0
�
training/Adam/ReadVariableOp_24ReadVariableOptraining/Adam/v_2_1!^training/Adam/AssignVariableOp_7*
dtype0*
_output_shapes	
:�
t
 training/Adam/AssignVariableOp_8AssignVariableOpbatch_normalization_1/gammatraining/Adam/sub_10*
dtype0
�
training/Adam/ReadVariableOp_25ReadVariableOpbatch_normalization_1/gamma!^training/Adam/AssignVariableOp_8*
dtype0*
_output_shapes	
:�
c
training/Adam/ReadVariableOp_26ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
t
#training/Adam/mul_16/ReadVariableOpReadVariableOptraining/Adam/m_3_1*
dtype0*
_output_shapes	
:�
�
training/Adam/mul_16Multraining/Adam/ReadVariableOp_26#training/Adam/mul_16/ReadVariableOp*
T0*
_output_shapes	
:�
c
training/Adam/ReadVariableOp_27ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
[
training/Adam/sub_11/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
u
training/Adam/sub_11Subtraining/Adam/sub_11/xtraining/Adam/ReadVariableOp_27*
T0*
_output_shapes
: 
x
training/Adam/mul_17Multraining/Adam/sub_11training/Adam/gradients/AddN_18*
_output_shapes	
:�*
T0
o
training/Adam/add_10AddV2training/Adam/mul_16training/Adam/mul_17*
T0*
_output_shapes	
:�
c
training/Adam/ReadVariableOp_28ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
t
#training/Adam/mul_18/ReadVariableOpReadVariableOptraining/Adam/v_3_1*
dtype0*
_output_shapes	
:�
�
training/Adam/mul_18Multraining/Adam/ReadVariableOp_28#training/Adam/mul_18/ReadVariableOp*
T0*
_output_shapes	
:�
c
training/Adam/ReadVariableOp_29ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
[
training/Adam/sub_12/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
u
training/Adam/sub_12Subtraining/Adam/sub_12/xtraining/Adam/ReadVariableOp_29*
_output_shapes
: *
T0
g
training/Adam/Square_3Squaretraining/Adam/gradients/AddN_18*
T0*
_output_shapes	
:�
o
training/Adam/mul_19Multraining/Adam/sub_12training/Adam/Square_3*
T0*
_output_shapes	
:�
o
training/Adam/add_11AddV2training/Adam/mul_18training/Adam/mul_19*
_output_shapes	
:�*
T0
j
training/Adam/mul_20Multraining/Adam/multraining/Adam/add_10*
T0*
_output_shapes	
:�
Z
training/Adam/Const_9Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_10Const*
valueB
 *  �*
dtype0*
_output_shapes
: 
�
%training/Adam/clip_by_value_4/MinimumMinimumtraining/Adam/add_11training/Adam/Const_10*
_output_shapes	
:�*
T0
�
training/Adam/clip_by_value_4Maximum%training/Adam/clip_by_value_4/Minimumtraining/Adam/Const_9*
_output_shapes	
:�*
T0
a
training/Adam/Sqrt_4Sqrttraining/Adam/clip_by_value_4*
T0*
_output_shapes	
:�
[
training/Adam/add_12/yConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
q
training/Adam/add_12AddV2training/Adam/Sqrt_4training/Adam/add_12/y*
_output_shapes	
:�*
T0
t
training/Adam/truediv_4RealDivtraining/Adam/mul_20training/Adam/add_12*
T0*
_output_shapes	
:�
w
training/Adam/ReadVariableOp_30ReadVariableOpbatch_normalization_1/beta*
dtype0*
_output_shapes	
:�
{
training/Adam/sub_13Subtraining/Adam/ReadVariableOp_30training/Adam/truediv_4*
T0*
_output_shapes	
:�
l
 training/Adam/AssignVariableOp_9AssignVariableOptraining/Adam/m_3_1training/Adam/add_10*
dtype0
�
training/Adam/ReadVariableOp_31ReadVariableOptraining/Adam/m_3_1!^training/Adam/AssignVariableOp_9*
dtype0*
_output_shapes	
:�
m
!training/Adam/AssignVariableOp_10AssignVariableOptraining/Adam/v_3_1training/Adam/add_11*
dtype0
�
training/Adam/ReadVariableOp_32ReadVariableOptraining/Adam/v_3_1"^training/Adam/AssignVariableOp_10*
dtype0*
_output_shapes	
:�
t
!training/Adam/AssignVariableOp_11AssignVariableOpbatch_normalization_1/betatraining/Adam/sub_13*
dtype0
�
training/Adam/ReadVariableOp_33ReadVariableOpbatch_normalization_1/beta"^training/Adam/AssignVariableOp_11*
dtype0*
_output_shapes	
:�
c
training/Adam/ReadVariableOp_34ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
y
#training/Adam/mul_21/ReadVariableOpReadVariableOptraining/Adam/m_4_1*
dtype0* 
_output_shapes
:
��
�
training/Adam/mul_21Multraining/Adam/ReadVariableOp_34#training/Adam/mul_21/ReadVariableOp*
T0* 
_output_shapes
:
��
c
training/Adam/ReadVariableOp_35ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
[
training/Adam/sub_14/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
u
training/Adam/sub_14Subtraining/Adam/sub_14/xtraining/Adam/ReadVariableOp_35*
_output_shapes
: *
T0
�
training/Adam/mul_22Multraining/Adam/sub_144training/Adam/gradients/dense_2/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:
��
t
training/Adam/add_13AddV2training/Adam/mul_21training/Adam/mul_22*
T0* 
_output_shapes
:
��
c
training/Adam/ReadVariableOp_36ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
y
#training/Adam/mul_23/ReadVariableOpReadVariableOptraining/Adam/v_4_1*
dtype0* 
_output_shapes
:
��
�
training/Adam/mul_23Multraining/Adam/ReadVariableOp_36#training/Adam/mul_23/ReadVariableOp*
T0* 
_output_shapes
:
��
c
training/Adam/ReadVariableOp_37ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
[
training/Adam/sub_15/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
u
training/Adam/sub_15Subtraining/Adam/sub_15/xtraining/Adam/ReadVariableOp_37*
T0*
_output_shapes
: 
�
training/Adam/Square_4Square4training/Adam/gradients/dense_2/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:
��
t
training/Adam/mul_24Multraining/Adam/sub_15training/Adam/Square_4*
T0* 
_output_shapes
:
��
t
training/Adam/add_14AddV2training/Adam/mul_23training/Adam/mul_24*
T0* 
_output_shapes
:
��
o
training/Adam/mul_25Multraining/Adam/multraining/Adam/add_13*
T0* 
_output_shapes
:
��
[
training/Adam/Const_11Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_12Const*
valueB
 *  �*
dtype0*
_output_shapes
: 
�
%training/Adam/clip_by_value_5/MinimumMinimumtraining/Adam/add_14training/Adam/Const_12*
T0* 
_output_shapes
:
��
�
training/Adam/clip_by_value_5Maximum%training/Adam/clip_by_value_5/Minimumtraining/Adam/Const_11*
T0* 
_output_shapes
:
��
f
training/Adam/Sqrt_5Sqrttraining/Adam/clip_by_value_5*
T0* 
_output_shapes
:
��
[
training/Adam/add_15/yConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
v
training/Adam/add_15AddV2training/Adam/Sqrt_5training/Adam/add_15/y* 
_output_shapes
:
��*
T0
y
training/Adam/truediv_5RealDivtraining/Adam/mul_25training/Adam/add_15*
T0* 
_output_shapes
:
��
p
training/Adam/ReadVariableOp_38ReadVariableOpdense_2/kernel*
dtype0* 
_output_shapes
:
��
�
training/Adam/sub_16Subtraining/Adam/ReadVariableOp_38training/Adam/truediv_5* 
_output_shapes
:
��*
T0
m
!training/Adam/AssignVariableOp_12AssignVariableOptraining/Adam/m_4_1training/Adam/add_13*
dtype0
�
training/Adam/ReadVariableOp_39ReadVariableOptraining/Adam/m_4_1"^training/Adam/AssignVariableOp_12*
dtype0* 
_output_shapes
:
��
m
!training/Adam/AssignVariableOp_13AssignVariableOptraining/Adam/v_4_1training/Adam/add_14*
dtype0
�
training/Adam/ReadVariableOp_40ReadVariableOptraining/Adam/v_4_1"^training/Adam/AssignVariableOp_13*
dtype0* 
_output_shapes
:
��
h
!training/Adam/AssignVariableOp_14AssignVariableOpdense_2/kerneltraining/Adam/sub_16*
dtype0
�
training/Adam/ReadVariableOp_41ReadVariableOpdense_2/kernel"^training/Adam/AssignVariableOp_14*
dtype0* 
_output_shapes
:
��
c
training/Adam/ReadVariableOp_42ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
t
#training/Adam/mul_26/ReadVariableOpReadVariableOptraining/Adam/m_5_1*
dtype0*
_output_shapes	
:�
�
training/Adam/mul_26Multraining/Adam/ReadVariableOp_42#training/Adam/mul_26/ReadVariableOp*
T0*
_output_shapes	
:�
c
training/Adam/ReadVariableOp_43ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
[
training/Adam/sub_17/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
u
training/Adam/sub_17Subtraining/Adam/sub_17/xtraining/Adam/ReadVariableOp_43*
_output_shapes
: *
T0
�
training/Adam/mul_27Multraining/Adam/sub_178training/Adam/gradients/dense_2/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:�
o
training/Adam/add_16AddV2training/Adam/mul_26training/Adam/mul_27*
T0*
_output_shapes	
:�
c
training/Adam/ReadVariableOp_44ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
t
#training/Adam/mul_28/ReadVariableOpReadVariableOptraining/Adam/v_5_1*
dtype0*
_output_shapes	
:�
�
training/Adam/mul_28Multraining/Adam/ReadVariableOp_44#training/Adam/mul_28/ReadVariableOp*
T0*
_output_shapes	
:�
c
training/Adam/ReadVariableOp_45ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
[
training/Adam/sub_18/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
u
training/Adam/sub_18Subtraining/Adam/sub_18/xtraining/Adam/ReadVariableOp_45*
_output_shapes
: *
T0
�
training/Adam/Square_5Square8training/Adam/gradients/dense_2/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:�
o
training/Adam/mul_29Multraining/Adam/sub_18training/Adam/Square_5*
T0*
_output_shapes	
:�
o
training/Adam/add_17AddV2training/Adam/mul_28training/Adam/mul_29*
T0*
_output_shapes	
:�
j
training/Adam/mul_30Multraining/Adam/multraining/Adam/add_16*
_output_shapes	
:�*
T0
[
training/Adam/Const_13Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_14Const*
valueB
 *  �*
dtype0*
_output_shapes
: 
�
%training/Adam/clip_by_value_6/MinimumMinimumtraining/Adam/add_17training/Adam/Const_14*
_output_shapes	
:�*
T0
�
training/Adam/clip_by_value_6Maximum%training/Adam/clip_by_value_6/Minimumtraining/Adam/Const_13*
_output_shapes	
:�*
T0
a
training/Adam/Sqrt_6Sqrttraining/Adam/clip_by_value_6*
T0*
_output_shapes	
:�
[
training/Adam/add_18/yConst*
dtype0*
_output_shapes
: *
valueB
 *���3
q
training/Adam/add_18AddV2training/Adam/Sqrt_6training/Adam/add_18/y*
_output_shapes	
:�*
T0
t
training/Adam/truediv_6RealDivtraining/Adam/mul_30training/Adam/add_18*
T0*
_output_shapes	
:�
i
training/Adam/ReadVariableOp_46ReadVariableOpdense_2/bias*
dtype0*
_output_shapes	
:�
{
training/Adam/sub_19Subtraining/Adam/ReadVariableOp_46training/Adam/truediv_6*
_output_shapes	
:�*
T0
m
!training/Adam/AssignVariableOp_15AssignVariableOptraining/Adam/m_5_1training/Adam/add_16*
dtype0
�
training/Adam/ReadVariableOp_47ReadVariableOptraining/Adam/m_5_1"^training/Adam/AssignVariableOp_15*
dtype0*
_output_shapes	
:�
m
!training/Adam/AssignVariableOp_16AssignVariableOptraining/Adam/v_5_1training/Adam/add_17*
dtype0
�
training/Adam/ReadVariableOp_48ReadVariableOptraining/Adam/v_5_1"^training/Adam/AssignVariableOp_16*
dtype0*
_output_shapes	
:�
f
!training/Adam/AssignVariableOp_17AssignVariableOpdense_2/biastraining/Adam/sub_19*
dtype0
�
training/Adam/ReadVariableOp_49ReadVariableOpdense_2/bias"^training/Adam/AssignVariableOp_17*
dtype0*
_output_shapes	
:�
c
training/Adam/ReadVariableOp_50ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
t
#training/Adam/mul_31/ReadVariableOpReadVariableOptraining/Adam/m_6_1*
dtype0*
_output_shapes	
:�
�
training/Adam/mul_31Multraining/Adam/ReadVariableOp_50#training/Adam/mul_31/ReadVariableOp*
T0*
_output_shapes	
:�
c
training/Adam/ReadVariableOp_51ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
[
training/Adam/sub_20/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
u
training/Adam/sub_20Subtraining/Adam/sub_20/xtraining/Adam/ReadVariableOp_51*
T0*
_output_shapes
: 
x
training/Adam/mul_32Multraining/Adam/sub_20training/Adam/gradients/AddN_15*
T0*
_output_shapes	
:�
o
training/Adam/add_19AddV2training/Adam/mul_31training/Adam/mul_32*
T0*
_output_shapes	
:�
c
training/Adam/ReadVariableOp_52ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
t
#training/Adam/mul_33/ReadVariableOpReadVariableOptraining/Adam/v_6_1*
dtype0*
_output_shapes	
:�
�
training/Adam/mul_33Multraining/Adam/ReadVariableOp_52#training/Adam/mul_33/ReadVariableOp*
_output_shapes	
:�*
T0
c
training/Adam/ReadVariableOp_53ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
[
training/Adam/sub_21/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
u
training/Adam/sub_21Subtraining/Adam/sub_21/xtraining/Adam/ReadVariableOp_53*
_output_shapes
: *
T0
g
training/Adam/Square_6Squaretraining/Adam/gradients/AddN_15*
T0*
_output_shapes	
:�
o
training/Adam/mul_34Multraining/Adam/sub_21training/Adam/Square_6*
T0*
_output_shapes	
:�
o
training/Adam/add_20AddV2training/Adam/mul_33training/Adam/mul_34*
T0*
_output_shapes	
:�
j
training/Adam/mul_35Multraining/Adam/multraining/Adam/add_19*
T0*
_output_shapes	
:�
[
training/Adam/Const_15Const*
dtype0*
_output_shapes
: *
valueB
 *    
[
training/Adam/Const_16Const*
valueB
 *  �*
dtype0*
_output_shapes
: 
�
%training/Adam/clip_by_value_7/MinimumMinimumtraining/Adam/add_20training/Adam/Const_16*
T0*
_output_shapes	
:�
�
training/Adam/clip_by_value_7Maximum%training/Adam/clip_by_value_7/Minimumtraining/Adam/Const_15*
T0*
_output_shapes	
:�
a
training/Adam/Sqrt_7Sqrttraining/Adam/clip_by_value_7*
T0*
_output_shapes	
:�
[
training/Adam/add_21/yConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
q
training/Adam/add_21AddV2training/Adam/Sqrt_7training/Adam/add_21/y*
T0*
_output_shapes	
:�
t
training/Adam/truediv_7RealDivtraining/Adam/mul_35training/Adam/add_21*
T0*
_output_shapes	
:�
x
training/Adam/ReadVariableOp_54ReadVariableOpbatch_normalization_2/gamma*
dtype0*
_output_shapes	
:�
{
training/Adam/sub_22Subtraining/Adam/ReadVariableOp_54training/Adam/truediv_7*
_output_shapes	
:�*
T0
m
!training/Adam/AssignVariableOp_18AssignVariableOptraining/Adam/m_6_1training/Adam/add_19*
dtype0
�
training/Adam/ReadVariableOp_55ReadVariableOptraining/Adam/m_6_1"^training/Adam/AssignVariableOp_18*
dtype0*
_output_shapes	
:�
m
!training/Adam/AssignVariableOp_19AssignVariableOptraining/Adam/v_6_1training/Adam/add_20*
dtype0
�
training/Adam/ReadVariableOp_56ReadVariableOptraining/Adam/v_6_1"^training/Adam/AssignVariableOp_19*
dtype0*
_output_shapes	
:�
u
!training/Adam/AssignVariableOp_20AssignVariableOpbatch_normalization_2/gammatraining/Adam/sub_22*
dtype0
�
training/Adam/ReadVariableOp_57ReadVariableOpbatch_normalization_2/gamma"^training/Adam/AssignVariableOp_20*
dtype0*
_output_shapes	
:�
c
training/Adam/ReadVariableOp_58ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
t
#training/Adam/mul_36/ReadVariableOpReadVariableOptraining/Adam/m_7_1*
dtype0*
_output_shapes	
:�
�
training/Adam/mul_36Multraining/Adam/ReadVariableOp_58#training/Adam/mul_36/ReadVariableOp*
T0*
_output_shapes	
:�
c
training/Adam/ReadVariableOp_59ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
[
training/Adam/sub_23/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
u
training/Adam/sub_23Subtraining/Adam/sub_23/xtraining/Adam/ReadVariableOp_59*
T0*
_output_shapes
: 
x
training/Adam/mul_37Multraining/Adam/sub_23training/Adam/gradients/AddN_13*
_output_shapes	
:�*
T0
o
training/Adam/add_22AddV2training/Adam/mul_36training/Adam/mul_37*
T0*
_output_shapes	
:�
c
training/Adam/ReadVariableOp_60ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
t
#training/Adam/mul_38/ReadVariableOpReadVariableOptraining/Adam/v_7_1*
dtype0*
_output_shapes	
:�
�
training/Adam/mul_38Multraining/Adam/ReadVariableOp_60#training/Adam/mul_38/ReadVariableOp*
T0*
_output_shapes	
:�
c
training/Adam/ReadVariableOp_61ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
[
training/Adam/sub_24/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
u
training/Adam/sub_24Subtraining/Adam/sub_24/xtraining/Adam/ReadVariableOp_61*
T0*
_output_shapes
: 
g
training/Adam/Square_7Squaretraining/Adam/gradients/AddN_13*
T0*
_output_shapes	
:�
o
training/Adam/mul_39Multraining/Adam/sub_24training/Adam/Square_7*
T0*
_output_shapes	
:�
o
training/Adam/add_23AddV2training/Adam/mul_38training/Adam/mul_39*
T0*
_output_shapes	
:�
j
training/Adam/mul_40Multraining/Adam/multraining/Adam/add_22*
_output_shapes	
:�*
T0
[
training/Adam/Const_17Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_18Const*
valueB
 *  �*
dtype0*
_output_shapes
: 
�
%training/Adam/clip_by_value_8/MinimumMinimumtraining/Adam/add_23training/Adam/Const_18*
T0*
_output_shapes	
:�
�
training/Adam/clip_by_value_8Maximum%training/Adam/clip_by_value_8/Minimumtraining/Adam/Const_17*
_output_shapes	
:�*
T0
a
training/Adam/Sqrt_8Sqrttraining/Adam/clip_by_value_8*
T0*
_output_shapes	
:�
[
training/Adam/add_24/yConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
q
training/Adam/add_24AddV2training/Adam/Sqrt_8training/Adam/add_24/y*
_output_shapes	
:�*
T0
t
training/Adam/truediv_8RealDivtraining/Adam/mul_40training/Adam/add_24*
_output_shapes	
:�*
T0
w
training/Adam/ReadVariableOp_62ReadVariableOpbatch_normalization_2/beta*
dtype0*
_output_shapes	
:�
{
training/Adam/sub_25Subtraining/Adam/ReadVariableOp_62training/Adam/truediv_8*
T0*
_output_shapes	
:�
m
!training/Adam/AssignVariableOp_21AssignVariableOptraining/Adam/m_7_1training/Adam/add_22*
dtype0
�
training/Adam/ReadVariableOp_63ReadVariableOptraining/Adam/m_7_1"^training/Adam/AssignVariableOp_21*
dtype0*
_output_shapes	
:�
m
!training/Adam/AssignVariableOp_22AssignVariableOptraining/Adam/v_7_1training/Adam/add_23*
dtype0
�
training/Adam/ReadVariableOp_64ReadVariableOptraining/Adam/v_7_1"^training/Adam/AssignVariableOp_22*
dtype0*
_output_shapes	
:�
t
!training/Adam/AssignVariableOp_23AssignVariableOpbatch_normalization_2/betatraining/Adam/sub_25*
dtype0
�
training/Adam/ReadVariableOp_65ReadVariableOpbatch_normalization_2/beta"^training/Adam/AssignVariableOp_23*
dtype0*
_output_shapes	
:�
c
training/Adam/ReadVariableOp_66ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
y
#training/Adam/mul_41/ReadVariableOpReadVariableOptraining/Adam/m_8_1*
dtype0* 
_output_shapes
:
��
�
training/Adam/mul_41Multraining/Adam/ReadVariableOp_66#training/Adam/mul_41/ReadVariableOp*
T0* 
_output_shapes
:
��
c
training/Adam/ReadVariableOp_67ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
[
training/Adam/sub_26/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
u
training/Adam/sub_26Subtraining/Adam/sub_26/xtraining/Adam/ReadVariableOp_67*
T0*
_output_shapes
: 
�
training/Adam/mul_42Multraining/Adam/sub_264training/Adam/gradients/dense_3/MatMul_grad/MatMul_1* 
_output_shapes
:
��*
T0
t
training/Adam/add_25AddV2training/Adam/mul_41training/Adam/mul_42*
T0* 
_output_shapes
:
��
c
training/Adam/ReadVariableOp_68ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
y
#training/Adam/mul_43/ReadVariableOpReadVariableOptraining/Adam/v_8_1*
dtype0* 
_output_shapes
:
��
�
training/Adam/mul_43Multraining/Adam/ReadVariableOp_68#training/Adam/mul_43/ReadVariableOp*
T0* 
_output_shapes
:
��
c
training/Adam/ReadVariableOp_69ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
[
training/Adam/sub_27/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
u
training/Adam/sub_27Subtraining/Adam/sub_27/xtraining/Adam/ReadVariableOp_69*
T0*
_output_shapes
: 
�
training/Adam/Square_8Square4training/Adam/gradients/dense_3/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:
��
t
training/Adam/mul_44Multraining/Adam/sub_27training/Adam/Square_8*
T0* 
_output_shapes
:
��
t
training/Adam/add_26AddV2training/Adam/mul_43training/Adam/mul_44*
T0* 
_output_shapes
:
��
o
training/Adam/mul_45Multraining/Adam/multraining/Adam/add_25*
T0* 
_output_shapes
:
��
[
training/Adam/Const_19Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_20Const*
valueB
 *  �*
dtype0*
_output_shapes
: 
�
%training/Adam/clip_by_value_9/MinimumMinimumtraining/Adam/add_26training/Adam/Const_20* 
_output_shapes
:
��*
T0
�
training/Adam/clip_by_value_9Maximum%training/Adam/clip_by_value_9/Minimumtraining/Adam/Const_19* 
_output_shapes
:
��*
T0
f
training/Adam/Sqrt_9Sqrttraining/Adam/clip_by_value_9*
T0* 
_output_shapes
:
��
[
training/Adam/add_27/yConst*
dtype0*
_output_shapes
: *
valueB
 *���3
v
training/Adam/add_27AddV2training/Adam/Sqrt_9training/Adam/add_27/y* 
_output_shapes
:
��*
T0
y
training/Adam/truediv_9RealDivtraining/Adam/mul_45training/Adam/add_27*
T0* 
_output_shapes
:
��
p
training/Adam/ReadVariableOp_70ReadVariableOpdense_3/kernel*
dtype0* 
_output_shapes
:
��
�
training/Adam/sub_28Subtraining/Adam/ReadVariableOp_70training/Adam/truediv_9*
T0* 
_output_shapes
:
��
m
!training/Adam/AssignVariableOp_24AssignVariableOptraining/Adam/m_8_1training/Adam/add_25*
dtype0
�
training/Adam/ReadVariableOp_71ReadVariableOptraining/Adam/m_8_1"^training/Adam/AssignVariableOp_24*
dtype0* 
_output_shapes
:
��
m
!training/Adam/AssignVariableOp_25AssignVariableOptraining/Adam/v_8_1training/Adam/add_26*
dtype0
�
training/Adam/ReadVariableOp_72ReadVariableOptraining/Adam/v_8_1"^training/Adam/AssignVariableOp_25*
dtype0* 
_output_shapes
:
��
h
!training/Adam/AssignVariableOp_26AssignVariableOpdense_3/kerneltraining/Adam/sub_28*
dtype0
�
training/Adam/ReadVariableOp_73ReadVariableOpdense_3/kernel"^training/Adam/AssignVariableOp_26*
dtype0* 
_output_shapes
:
��
c
training/Adam/ReadVariableOp_74ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
t
#training/Adam/mul_46/ReadVariableOpReadVariableOptraining/Adam/m_9_1*
dtype0*
_output_shapes	
:�
�
training/Adam/mul_46Multraining/Adam/ReadVariableOp_74#training/Adam/mul_46/ReadVariableOp*
T0*
_output_shapes	
:�
c
training/Adam/ReadVariableOp_75ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
[
training/Adam/sub_29/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
u
training/Adam/sub_29Subtraining/Adam/sub_29/xtraining/Adam/ReadVariableOp_75*
T0*
_output_shapes
: 
�
training/Adam/mul_47Multraining/Adam/sub_298training/Adam/gradients/dense_3/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�*
T0
o
training/Adam/add_28AddV2training/Adam/mul_46training/Adam/mul_47*
T0*
_output_shapes	
:�
c
training/Adam/ReadVariableOp_76ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
t
#training/Adam/mul_48/ReadVariableOpReadVariableOptraining/Adam/v_9_1*
dtype0*
_output_shapes	
:�
�
training/Adam/mul_48Multraining/Adam/ReadVariableOp_76#training/Adam/mul_48/ReadVariableOp*
T0*
_output_shapes	
:�
c
training/Adam/ReadVariableOp_77ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
[
training/Adam/sub_30/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
u
training/Adam/sub_30Subtraining/Adam/sub_30/xtraining/Adam/ReadVariableOp_77*
T0*
_output_shapes
: 
�
training/Adam/Square_9Square8training/Adam/gradients/dense_3/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:�
o
training/Adam/mul_49Multraining/Adam/sub_30training/Adam/Square_9*
T0*
_output_shapes	
:�
o
training/Adam/add_29AddV2training/Adam/mul_48training/Adam/mul_49*
_output_shapes	
:�*
T0
j
training/Adam/mul_50Multraining/Adam/multraining/Adam/add_28*
T0*
_output_shapes	
:�
[
training/Adam/Const_21Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_22Const*
valueB
 *  �*
dtype0*
_output_shapes
: 
�
&training/Adam/clip_by_value_10/MinimumMinimumtraining/Adam/add_29training/Adam/Const_22*
T0*
_output_shapes	
:�
�
training/Adam/clip_by_value_10Maximum&training/Adam/clip_by_value_10/Minimumtraining/Adam/Const_21*
_output_shapes	
:�*
T0
c
training/Adam/Sqrt_10Sqrttraining/Adam/clip_by_value_10*
T0*
_output_shapes	
:�
[
training/Adam/add_30/yConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
r
training/Adam/add_30AddV2training/Adam/Sqrt_10training/Adam/add_30/y*
T0*
_output_shapes	
:�
u
training/Adam/truediv_10RealDivtraining/Adam/mul_50training/Adam/add_30*
T0*
_output_shapes	
:�
i
training/Adam/ReadVariableOp_78ReadVariableOpdense_3/bias*
dtype0*
_output_shapes	
:�
|
training/Adam/sub_31Subtraining/Adam/ReadVariableOp_78training/Adam/truediv_10*
T0*
_output_shapes	
:�
m
!training/Adam/AssignVariableOp_27AssignVariableOptraining/Adam/m_9_1training/Adam/add_28*
dtype0
�
training/Adam/ReadVariableOp_79ReadVariableOptraining/Adam/m_9_1"^training/Adam/AssignVariableOp_27*
dtype0*
_output_shapes	
:�
m
!training/Adam/AssignVariableOp_28AssignVariableOptraining/Adam/v_9_1training/Adam/add_29*
dtype0
�
training/Adam/ReadVariableOp_80ReadVariableOptraining/Adam/v_9_1"^training/Adam/AssignVariableOp_28*
dtype0*
_output_shapes	
:�
f
!training/Adam/AssignVariableOp_29AssignVariableOpdense_3/biastraining/Adam/sub_31*
dtype0
�
training/Adam/ReadVariableOp_81ReadVariableOpdense_3/bias"^training/Adam/AssignVariableOp_29*
dtype0*
_output_shapes	
:�
c
training/Adam/ReadVariableOp_82ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
u
#training/Adam/mul_51/ReadVariableOpReadVariableOptraining/Adam/m_10_1*
dtype0*
_output_shapes	
:�
�
training/Adam/mul_51Multraining/Adam/ReadVariableOp_82#training/Adam/mul_51/ReadVariableOp*
T0*
_output_shapes	
:�
c
training/Adam/ReadVariableOp_83ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
[
training/Adam/sub_32/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
u
training/Adam/sub_32Subtraining/Adam/sub_32/xtraining/Adam/ReadVariableOp_83*
T0*
_output_shapes
: 
w
training/Adam/mul_52Multraining/Adam/sub_32training/Adam/gradients/AddN_9*
_output_shapes	
:�*
T0
o
training/Adam/add_31AddV2training/Adam/mul_51training/Adam/mul_52*
_output_shapes	
:�*
T0
c
training/Adam/ReadVariableOp_84ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
u
#training/Adam/mul_53/ReadVariableOpReadVariableOptraining/Adam/v_10_1*
dtype0*
_output_shapes	
:�
�
training/Adam/mul_53Multraining/Adam/ReadVariableOp_84#training/Adam/mul_53/ReadVariableOp*
T0*
_output_shapes	
:�
c
training/Adam/ReadVariableOp_85ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
[
training/Adam/sub_33/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
u
training/Adam/sub_33Subtraining/Adam/sub_33/xtraining/Adam/ReadVariableOp_85*
T0*
_output_shapes
: 
g
training/Adam/Square_10Squaretraining/Adam/gradients/AddN_9*
T0*
_output_shapes	
:�
p
training/Adam/mul_54Multraining/Adam/sub_33training/Adam/Square_10*
T0*
_output_shapes	
:�
o
training/Adam/add_32AddV2training/Adam/mul_53training/Adam/mul_54*
T0*
_output_shapes	
:�
j
training/Adam/mul_55Multraining/Adam/multraining/Adam/add_31*
_output_shapes	
:�*
T0
[
training/Adam/Const_23Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_24Const*
valueB
 *  �*
dtype0*
_output_shapes
: 
�
&training/Adam/clip_by_value_11/MinimumMinimumtraining/Adam/add_32training/Adam/Const_24*
_output_shapes	
:�*
T0
�
training/Adam/clip_by_value_11Maximum&training/Adam/clip_by_value_11/Minimumtraining/Adam/Const_23*
T0*
_output_shapes	
:�
c
training/Adam/Sqrt_11Sqrttraining/Adam/clip_by_value_11*
T0*
_output_shapes	
:�
[
training/Adam/add_33/yConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
r
training/Adam/add_33AddV2training/Adam/Sqrt_11training/Adam/add_33/y*
T0*
_output_shapes	
:�
u
training/Adam/truediv_11RealDivtraining/Adam/mul_55training/Adam/add_33*
_output_shapes	
:�*
T0
x
training/Adam/ReadVariableOp_86ReadVariableOpbatch_normalization_3/gamma*
dtype0*
_output_shapes	
:�
|
training/Adam/sub_34Subtraining/Adam/ReadVariableOp_86training/Adam/truediv_11*
T0*
_output_shapes	
:�
n
!training/Adam/AssignVariableOp_30AssignVariableOptraining/Adam/m_10_1training/Adam/add_31*
dtype0
�
training/Adam/ReadVariableOp_87ReadVariableOptraining/Adam/m_10_1"^training/Adam/AssignVariableOp_30*
dtype0*
_output_shapes	
:�
n
!training/Adam/AssignVariableOp_31AssignVariableOptraining/Adam/v_10_1training/Adam/add_32*
dtype0
�
training/Adam/ReadVariableOp_88ReadVariableOptraining/Adam/v_10_1"^training/Adam/AssignVariableOp_31*
dtype0*
_output_shapes	
:�
u
!training/Adam/AssignVariableOp_32AssignVariableOpbatch_normalization_3/gammatraining/Adam/sub_34*
dtype0
�
training/Adam/ReadVariableOp_89ReadVariableOpbatch_normalization_3/gamma"^training/Adam/AssignVariableOp_32*
dtype0*
_output_shapes	
:�
c
training/Adam/ReadVariableOp_90ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
u
#training/Adam/mul_56/ReadVariableOpReadVariableOptraining/Adam/m_11_1*
dtype0*
_output_shapes	
:�
�
training/Adam/mul_56Multraining/Adam/ReadVariableOp_90#training/Adam/mul_56/ReadVariableOp*
T0*
_output_shapes	
:�
c
training/Adam/ReadVariableOp_91ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
[
training/Adam/sub_35/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
u
training/Adam/sub_35Subtraining/Adam/sub_35/xtraining/Adam/ReadVariableOp_91*
_output_shapes
: *
T0
w
training/Adam/mul_57Multraining/Adam/sub_35training/Adam/gradients/AddN_7*
T0*
_output_shapes	
:�
o
training/Adam/add_34AddV2training/Adam/mul_56training/Adam/mul_57*
T0*
_output_shapes	
:�
c
training/Adam/ReadVariableOp_92ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
u
#training/Adam/mul_58/ReadVariableOpReadVariableOptraining/Adam/v_11_1*
dtype0*
_output_shapes	
:�
�
training/Adam/mul_58Multraining/Adam/ReadVariableOp_92#training/Adam/mul_58/ReadVariableOp*
T0*
_output_shapes	
:�
c
training/Adam/ReadVariableOp_93ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
[
training/Adam/sub_36/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
u
training/Adam/sub_36Subtraining/Adam/sub_36/xtraining/Adam/ReadVariableOp_93*
_output_shapes
: *
T0
g
training/Adam/Square_11Squaretraining/Adam/gradients/AddN_7*
T0*
_output_shapes	
:�
p
training/Adam/mul_59Multraining/Adam/sub_36training/Adam/Square_11*
T0*
_output_shapes	
:�
o
training/Adam/add_35AddV2training/Adam/mul_58training/Adam/mul_59*
T0*
_output_shapes	
:�
j
training/Adam/mul_60Multraining/Adam/multraining/Adam/add_34*
_output_shapes	
:�*
T0
[
training/Adam/Const_25Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_26Const*
valueB
 *  �*
dtype0*
_output_shapes
: 
�
&training/Adam/clip_by_value_12/MinimumMinimumtraining/Adam/add_35training/Adam/Const_26*
_output_shapes	
:�*
T0
�
training/Adam/clip_by_value_12Maximum&training/Adam/clip_by_value_12/Minimumtraining/Adam/Const_25*
T0*
_output_shapes	
:�
c
training/Adam/Sqrt_12Sqrttraining/Adam/clip_by_value_12*
T0*
_output_shapes	
:�
[
training/Adam/add_36/yConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
r
training/Adam/add_36AddV2training/Adam/Sqrt_12training/Adam/add_36/y*
T0*
_output_shapes	
:�
u
training/Adam/truediv_12RealDivtraining/Adam/mul_60training/Adam/add_36*
T0*
_output_shapes	
:�
w
training/Adam/ReadVariableOp_94ReadVariableOpbatch_normalization_3/beta*
dtype0*
_output_shapes	
:�
|
training/Adam/sub_37Subtraining/Adam/ReadVariableOp_94training/Adam/truediv_12*
T0*
_output_shapes	
:�
n
!training/Adam/AssignVariableOp_33AssignVariableOptraining/Adam/m_11_1training/Adam/add_34*
dtype0
�
training/Adam/ReadVariableOp_95ReadVariableOptraining/Adam/m_11_1"^training/Adam/AssignVariableOp_33*
dtype0*
_output_shapes	
:�
n
!training/Adam/AssignVariableOp_34AssignVariableOptraining/Adam/v_11_1training/Adam/add_35*
dtype0
�
training/Adam/ReadVariableOp_96ReadVariableOptraining/Adam/v_11_1"^training/Adam/AssignVariableOp_34*
dtype0*
_output_shapes	
:�
t
!training/Adam/AssignVariableOp_35AssignVariableOpbatch_normalization_3/betatraining/Adam/sub_37*
dtype0
�
training/Adam/ReadVariableOp_97ReadVariableOpbatch_normalization_3/beta"^training/Adam/AssignVariableOp_35*
dtype0*
_output_shapes	
:�
c
training/Adam/ReadVariableOp_98ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
z
#training/Adam/mul_61/ReadVariableOpReadVariableOptraining/Adam/m_12_1*
dtype0* 
_output_shapes
:
��
�
training/Adam/mul_61Multraining/Adam/ReadVariableOp_98#training/Adam/mul_61/ReadVariableOp* 
_output_shapes
:
��*
T0
c
training/Adam/ReadVariableOp_99ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
[
training/Adam/sub_38/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
u
training/Adam/sub_38Subtraining/Adam/sub_38/xtraining/Adam/ReadVariableOp_99*
T0*
_output_shapes
: 
�
training/Adam/mul_62Multraining/Adam/sub_384training/Adam/gradients/dense_4/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:
��
t
training/Adam/add_37AddV2training/Adam/mul_61training/Adam/mul_62*
T0* 
_output_shapes
:
��
d
 training/Adam/ReadVariableOp_100ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
z
#training/Adam/mul_63/ReadVariableOpReadVariableOptraining/Adam/v_12_1*
dtype0* 
_output_shapes
:
��
�
training/Adam/mul_63Mul training/Adam/ReadVariableOp_100#training/Adam/mul_63/ReadVariableOp* 
_output_shapes
:
��*
T0
d
 training/Adam/ReadVariableOp_101ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
[
training/Adam/sub_39/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
v
training/Adam/sub_39Subtraining/Adam/sub_39/x training/Adam/ReadVariableOp_101*
T0*
_output_shapes
: 
�
training/Adam/Square_12Square4training/Adam/gradients/dense_4/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:
��
u
training/Adam/mul_64Multraining/Adam/sub_39training/Adam/Square_12*
T0* 
_output_shapes
:
��
t
training/Adam/add_38AddV2training/Adam/mul_63training/Adam/mul_64* 
_output_shapes
:
��*
T0
o
training/Adam/mul_65Multraining/Adam/multraining/Adam/add_37*
T0* 
_output_shapes
:
��
[
training/Adam/Const_27Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_28Const*
dtype0*
_output_shapes
: *
valueB
 *  �
�
&training/Adam/clip_by_value_13/MinimumMinimumtraining/Adam/add_38training/Adam/Const_28* 
_output_shapes
:
��*
T0
�
training/Adam/clip_by_value_13Maximum&training/Adam/clip_by_value_13/Minimumtraining/Adam/Const_27* 
_output_shapes
:
��*
T0
h
training/Adam/Sqrt_13Sqrttraining/Adam/clip_by_value_13*
T0* 
_output_shapes
:
��
[
training/Adam/add_39/yConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
w
training/Adam/add_39AddV2training/Adam/Sqrt_13training/Adam/add_39/y* 
_output_shapes
:
��*
T0
z
training/Adam/truediv_13RealDivtraining/Adam/mul_65training/Adam/add_39*
T0* 
_output_shapes
:
��
q
 training/Adam/ReadVariableOp_102ReadVariableOpdense_4/kernel*
dtype0* 
_output_shapes
:
��
�
training/Adam/sub_40Sub training/Adam/ReadVariableOp_102training/Adam/truediv_13*
T0* 
_output_shapes
:
��
n
!training/Adam/AssignVariableOp_36AssignVariableOptraining/Adam/m_12_1training/Adam/add_37*
dtype0
�
 training/Adam/ReadVariableOp_103ReadVariableOptraining/Adam/m_12_1"^training/Adam/AssignVariableOp_36*
dtype0* 
_output_shapes
:
��
n
!training/Adam/AssignVariableOp_37AssignVariableOptraining/Adam/v_12_1training/Adam/add_38*
dtype0
�
 training/Adam/ReadVariableOp_104ReadVariableOptraining/Adam/v_12_1"^training/Adam/AssignVariableOp_37*
dtype0* 
_output_shapes
:
��
h
!training/Adam/AssignVariableOp_38AssignVariableOpdense_4/kerneltraining/Adam/sub_40*
dtype0
�
 training/Adam/ReadVariableOp_105ReadVariableOpdense_4/kernel"^training/Adam/AssignVariableOp_38*
dtype0* 
_output_shapes
:
��
d
 training/Adam/ReadVariableOp_106ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
u
#training/Adam/mul_66/ReadVariableOpReadVariableOptraining/Adam/m_13_1*
dtype0*
_output_shapes	
:�
�
training/Adam/mul_66Mul training/Adam/ReadVariableOp_106#training/Adam/mul_66/ReadVariableOp*
T0*
_output_shapes	
:�
d
 training/Adam/ReadVariableOp_107ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
[
training/Adam/sub_41/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
v
training/Adam/sub_41Subtraining/Adam/sub_41/x training/Adam/ReadVariableOp_107*
T0*
_output_shapes
: 
�
training/Adam/mul_67Multraining/Adam/sub_418training/Adam/gradients/dense_4/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�*
T0
o
training/Adam/add_40AddV2training/Adam/mul_66training/Adam/mul_67*
_output_shapes	
:�*
T0
d
 training/Adam/ReadVariableOp_108ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
u
#training/Adam/mul_68/ReadVariableOpReadVariableOptraining/Adam/v_13_1*
dtype0*
_output_shapes	
:�
�
training/Adam/mul_68Mul training/Adam/ReadVariableOp_108#training/Adam/mul_68/ReadVariableOp*
T0*
_output_shapes	
:�
d
 training/Adam/ReadVariableOp_109ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
[
training/Adam/sub_42/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
v
training/Adam/sub_42Subtraining/Adam/sub_42/x training/Adam/ReadVariableOp_109*
T0*
_output_shapes
: 
�
training/Adam/Square_13Square8training/Adam/gradients/dense_4/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:�
p
training/Adam/mul_69Multraining/Adam/sub_42training/Adam/Square_13*
_output_shapes	
:�*
T0
o
training/Adam/add_41AddV2training/Adam/mul_68training/Adam/mul_69*
_output_shapes	
:�*
T0
j
training/Adam/mul_70Multraining/Adam/multraining/Adam/add_40*
T0*
_output_shapes	
:�
[
training/Adam/Const_29Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_30Const*
valueB
 *  �*
dtype0*
_output_shapes
: 
�
&training/Adam/clip_by_value_14/MinimumMinimumtraining/Adam/add_41training/Adam/Const_30*
T0*
_output_shapes	
:�
�
training/Adam/clip_by_value_14Maximum&training/Adam/clip_by_value_14/Minimumtraining/Adam/Const_29*
T0*
_output_shapes	
:�
c
training/Adam/Sqrt_14Sqrttraining/Adam/clip_by_value_14*
T0*
_output_shapes	
:�
[
training/Adam/add_42/yConst*
dtype0*
_output_shapes
: *
valueB
 *���3
r
training/Adam/add_42AddV2training/Adam/Sqrt_14training/Adam/add_42/y*
T0*
_output_shapes	
:�
u
training/Adam/truediv_14RealDivtraining/Adam/mul_70training/Adam/add_42*
T0*
_output_shapes	
:�
j
 training/Adam/ReadVariableOp_110ReadVariableOpdense_4/bias*
dtype0*
_output_shapes	
:�
}
training/Adam/sub_43Sub training/Adam/ReadVariableOp_110training/Adam/truediv_14*
T0*
_output_shapes	
:�
n
!training/Adam/AssignVariableOp_39AssignVariableOptraining/Adam/m_13_1training/Adam/add_40*
dtype0
�
 training/Adam/ReadVariableOp_111ReadVariableOptraining/Adam/m_13_1"^training/Adam/AssignVariableOp_39*
dtype0*
_output_shapes	
:�
n
!training/Adam/AssignVariableOp_40AssignVariableOptraining/Adam/v_13_1training/Adam/add_41*
dtype0
�
 training/Adam/ReadVariableOp_112ReadVariableOptraining/Adam/v_13_1"^training/Adam/AssignVariableOp_40*
dtype0*
_output_shapes	
:�
f
!training/Adam/AssignVariableOp_41AssignVariableOpdense_4/biastraining/Adam/sub_43*
dtype0
�
 training/Adam/ReadVariableOp_113ReadVariableOpdense_4/bias"^training/Adam/AssignVariableOp_41*
dtype0*
_output_shapes	
:�
d
 training/Adam/ReadVariableOp_114ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
u
#training/Adam/mul_71/ReadVariableOpReadVariableOptraining/Adam/m_14_1*
dtype0*
_output_shapes	
:�
�
training/Adam/mul_71Mul training/Adam/ReadVariableOp_114#training/Adam/mul_71/ReadVariableOp*
T0*
_output_shapes	
:�
d
 training/Adam/ReadVariableOp_115ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
[
training/Adam/sub_44/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
v
training/Adam/sub_44Subtraining/Adam/sub_44/x training/Adam/ReadVariableOp_115*
T0*
_output_shapes
: 
w
training/Adam/mul_72Multraining/Adam/sub_44training/Adam/gradients/AddN_3*
T0*
_output_shapes	
:�
o
training/Adam/add_43AddV2training/Adam/mul_71training/Adam/mul_72*
T0*
_output_shapes	
:�
d
 training/Adam/ReadVariableOp_116ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
u
#training/Adam/mul_73/ReadVariableOpReadVariableOptraining/Adam/v_14_1*
dtype0*
_output_shapes	
:�
�
training/Adam/mul_73Mul training/Adam/ReadVariableOp_116#training/Adam/mul_73/ReadVariableOp*
_output_shapes	
:�*
T0
d
 training/Adam/ReadVariableOp_117ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
[
training/Adam/sub_45/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
v
training/Adam/sub_45Subtraining/Adam/sub_45/x training/Adam/ReadVariableOp_117*
T0*
_output_shapes
: 
g
training/Adam/Square_14Squaretraining/Adam/gradients/AddN_3*
_output_shapes	
:�*
T0
p
training/Adam/mul_74Multraining/Adam/sub_45training/Adam/Square_14*
T0*
_output_shapes	
:�
o
training/Adam/add_44AddV2training/Adam/mul_73training/Adam/mul_74*
T0*
_output_shapes	
:�
j
training/Adam/mul_75Multraining/Adam/multraining/Adam/add_43*
T0*
_output_shapes	
:�
[
training/Adam/Const_31Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_32Const*
valueB
 *  �*
dtype0*
_output_shapes
: 
�
&training/Adam/clip_by_value_15/MinimumMinimumtraining/Adam/add_44training/Adam/Const_32*
T0*
_output_shapes	
:�
�
training/Adam/clip_by_value_15Maximum&training/Adam/clip_by_value_15/Minimumtraining/Adam/Const_31*
T0*
_output_shapes	
:�
c
training/Adam/Sqrt_15Sqrttraining/Adam/clip_by_value_15*
T0*
_output_shapes	
:�
[
training/Adam/add_45/yConst*
dtype0*
_output_shapes
: *
valueB
 *���3
r
training/Adam/add_45AddV2training/Adam/Sqrt_15training/Adam/add_45/y*
T0*
_output_shapes	
:�
u
training/Adam/truediv_15RealDivtraining/Adam/mul_75training/Adam/add_45*
_output_shapes	
:�*
T0
y
 training/Adam/ReadVariableOp_118ReadVariableOpbatch_normalization_4/gamma*
dtype0*
_output_shapes	
:�
}
training/Adam/sub_46Sub training/Adam/ReadVariableOp_118training/Adam/truediv_15*
_output_shapes	
:�*
T0
n
!training/Adam/AssignVariableOp_42AssignVariableOptraining/Adam/m_14_1training/Adam/add_43*
dtype0
�
 training/Adam/ReadVariableOp_119ReadVariableOptraining/Adam/m_14_1"^training/Adam/AssignVariableOp_42*
dtype0*
_output_shapes	
:�
n
!training/Adam/AssignVariableOp_43AssignVariableOptraining/Adam/v_14_1training/Adam/add_44*
dtype0
�
 training/Adam/ReadVariableOp_120ReadVariableOptraining/Adam/v_14_1"^training/Adam/AssignVariableOp_43*
dtype0*
_output_shapes	
:�
u
!training/Adam/AssignVariableOp_44AssignVariableOpbatch_normalization_4/gammatraining/Adam/sub_46*
dtype0
�
 training/Adam/ReadVariableOp_121ReadVariableOpbatch_normalization_4/gamma"^training/Adam/AssignVariableOp_44*
dtype0*
_output_shapes	
:�
d
 training/Adam/ReadVariableOp_122ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
u
#training/Adam/mul_76/ReadVariableOpReadVariableOptraining/Adam/m_15_1*
dtype0*
_output_shapes	
:�
�
training/Adam/mul_76Mul training/Adam/ReadVariableOp_122#training/Adam/mul_76/ReadVariableOp*
T0*
_output_shapes	
:�
d
 training/Adam/ReadVariableOp_123ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
[
training/Adam/sub_47/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
v
training/Adam/sub_47Subtraining/Adam/sub_47/x training/Adam/ReadVariableOp_123*
T0*
_output_shapes
: 
w
training/Adam/mul_77Multraining/Adam/sub_47training/Adam/gradients/AddN_1*
_output_shapes	
:�*
T0
o
training/Adam/add_46AddV2training/Adam/mul_76training/Adam/mul_77*
T0*
_output_shapes	
:�
d
 training/Adam/ReadVariableOp_124ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
u
#training/Adam/mul_78/ReadVariableOpReadVariableOptraining/Adam/v_15_1*
dtype0*
_output_shapes	
:�
�
training/Adam/mul_78Mul training/Adam/ReadVariableOp_124#training/Adam/mul_78/ReadVariableOp*
T0*
_output_shapes	
:�
d
 training/Adam/ReadVariableOp_125ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
[
training/Adam/sub_48/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
v
training/Adam/sub_48Subtraining/Adam/sub_48/x training/Adam/ReadVariableOp_125*
T0*
_output_shapes
: 
g
training/Adam/Square_15Squaretraining/Adam/gradients/AddN_1*
T0*
_output_shapes	
:�
p
training/Adam/mul_79Multraining/Adam/sub_48training/Adam/Square_15*
T0*
_output_shapes	
:�
o
training/Adam/add_47AddV2training/Adam/mul_78training/Adam/mul_79*
T0*
_output_shapes	
:�
j
training/Adam/mul_80Multraining/Adam/multraining/Adam/add_46*
_output_shapes	
:�*
T0
[
training/Adam/Const_33Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_34Const*
valueB
 *  �*
dtype0*
_output_shapes
: 
�
&training/Adam/clip_by_value_16/MinimumMinimumtraining/Adam/add_47training/Adam/Const_34*
T0*
_output_shapes	
:�
�
training/Adam/clip_by_value_16Maximum&training/Adam/clip_by_value_16/Minimumtraining/Adam/Const_33*
T0*
_output_shapes	
:�
c
training/Adam/Sqrt_16Sqrttraining/Adam/clip_by_value_16*
T0*
_output_shapes	
:�
[
training/Adam/add_48/yConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
r
training/Adam/add_48AddV2training/Adam/Sqrt_16training/Adam/add_48/y*
_output_shapes	
:�*
T0
u
training/Adam/truediv_16RealDivtraining/Adam/mul_80training/Adam/add_48*
T0*
_output_shapes	
:�
x
 training/Adam/ReadVariableOp_126ReadVariableOpbatch_normalization_4/beta*
dtype0*
_output_shapes	
:�
}
training/Adam/sub_49Sub training/Adam/ReadVariableOp_126training/Adam/truediv_16*
_output_shapes	
:�*
T0
n
!training/Adam/AssignVariableOp_45AssignVariableOptraining/Adam/m_15_1training/Adam/add_46*
dtype0
�
 training/Adam/ReadVariableOp_127ReadVariableOptraining/Adam/m_15_1"^training/Adam/AssignVariableOp_45*
dtype0*
_output_shapes	
:�
n
!training/Adam/AssignVariableOp_46AssignVariableOptraining/Adam/v_15_1training/Adam/add_47*
dtype0
�
 training/Adam/ReadVariableOp_128ReadVariableOptraining/Adam/v_15_1"^training/Adam/AssignVariableOp_46*
dtype0*
_output_shapes	
:�
t
!training/Adam/AssignVariableOp_47AssignVariableOpbatch_normalization_4/betatraining/Adam/sub_49*
dtype0
�
 training/Adam/ReadVariableOp_129ReadVariableOpbatch_normalization_4/beta"^training/Adam/AssignVariableOp_47*
dtype0*
_output_shapes	
:�
d
 training/Adam/ReadVariableOp_130ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
y
#training/Adam/mul_81/ReadVariableOpReadVariableOptraining/Adam/m_16_1*
dtype0*
_output_shapes
:	�+
�
training/Adam/mul_81Mul training/Adam/ReadVariableOp_130#training/Adam/mul_81/ReadVariableOp*
T0*
_output_shapes
:	�+
d
 training/Adam/ReadVariableOp_131ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
[
training/Adam/sub_50/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
v
training/Adam/sub_50Subtraining/Adam/sub_50/x training/Adam/ReadVariableOp_131*
T0*
_output_shapes
: 
�
training/Adam/mul_82Multraining/Adam/sub_504training/Adam/gradients/dense_5/MatMul_grad/MatMul_1*
T0*
_output_shapes
:	�+
s
training/Adam/add_49AddV2training/Adam/mul_81training/Adam/mul_82*
T0*
_output_shapes
:	�+
d
 training/Adam/ReadVariableOp_132ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
y
#training/Adam/mul_83/ReadVariableOpReadVariableOptraining/Adam/v_16_1*
dtype0*
_output_shapes
:	�+
�
training/Adam/mul_83Mul training/Adam/ReadVariableOp_132#training/Adam/mul_83/ReadVariableOp*
T0*
_output_shapes
:	�+
d
 training/Adam/ReadVariableOp_133ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
[
training/Adam/sub_51/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
v
training/Adam/sub_51Subtraining/Adam/sub_51/x training/Adam/ReadVariableOp_133*
T0*
_output_shapes
: 
�
training/Adam/Square_16Square4training/Adam/gradients/dense_5/MatMul_grad/MatMul_1*
T0*
_output_shapes
:	�+
t
training/Adam/mul_84Multraining/Adam/sub_51training/Adam/Square_16*
T0*
_output_shapes
:	�+
s
training/Adam/add_50AddV2training/Adam/mul_83training/Adam/mul_84*
_output_shapes
:	�+*
T0
n
training/Adam/mul_85Multraining/Adam/multraining/Adam/add_49*
_output_shapes
:	�+*
T0
[
training/Adam/Const_35Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_36Const*
dtype0*
_output_shapes
: *
valueB
 *  �
�
&training/Adam/clip_by_value_17/MinimumMinimumtraining/Adam/add_50training/Adam/Const_36*
T0*
_output_shapes
:	�+
�
training/Adam/clip_by_value_17Maximum&training/Adam/clip_by_value_17/Minimumtraining/Adam/Const_35*
T0*
_output_shapes
:	�+
g
training/Adam/Sqrt_17Sqrttraining/Adam/clip_by_value_17*
T0*
_output_shapes
:	�+
[
training/Adam/add_51/yConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
v
training/Adam/add_51AddV2training/Adam/Sqrt_17training/Adam/add_51/y*
T0*
_output_shapes
:	�+
y
training/Adam/truediv_17RealDivtraining/Adam/mul_85training/Adam/add_51*
_output_shapes
:	�+*
T0
p
 training/Adam/ReadVariableOp_134ReadVariableOpdense_5/kernel*
dtype0*
_output_shapes
:	�+
�
training/Adam/sub_52Sub training/Adam/ReadVariableOp_134training/Adam/truediv_17*
_output_shapes
:	�+*
T0
n
!training/Adam/AssignVariableOp_48AssignVariableOptraining/Adam/m_16_1training/Adam/add_49*
dtype0
�
 training/Adam/ReadVariableOp_135ReadVariableOptraining/Adam/m_16_1"^training/Adam/AssignVariableOp_48*
dtype0*
_output_shapes
:	�+
n
!training/Adam/AssignVariableOp_49AssignVariableOptraining/Adam/v_16_1training/Adam/add_50*
dtype0
�
 training/Adam/ReadVariableOp_136ReadVariableOptraining/Adam/v_16_1"^training/Adam/AssignVariableOp_49*
dtype0*
_output_shapes
:	�+
h
!training/Adam/AssignVariableOp_50AssignVariableOpdense_5/kerneltraining/Adam/sub_52*
dtype0
�
 training/Adam/ReadVariableOp_137ReadVariableOpdense_5/kernel"^training/Adam/AssignVariableOp_50*
dtype0*
_output_shapes
:	�+
d
 training/Adam/ReadVariableOp_138ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
t
#training/Adam/mul_86/ReadVariableOpReadVariableOptraining/Adam/m_17_1*
dtype0*
_output_shapes
:+
�
training/Adam/mul_86Mul training/Adam/ReadVariableOp_138#training/Adam/mul_86/ReadVariableOp*
T0*
_output_shapes
:+
d
 training/Adam/ReadVariableOp_139ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
[
training/Adam/sub_53/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
v
training/Adam/sub_53Subtraining/Adam/sub_53/x training/Adam/ReadVariableOp_139*
_output_shapes
: *
T0
�
training/Adam/mul_87Multraining/Adam/sub_538training/Adam/gradients/dense_5/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:+
n
training/Adam/add_52AddV2training/Adam/mul_86training/Adam/mul_87*
_output_shapes
:+*
T0
d
 training/Adam/ReadVariableOp_140ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
t
#training/Adam/mul_88/ReadVariableOpReadVariableOptraining/Adam/v_17_1*
dtype0*
_output_shapes
:+
�
training/Adam/mul_88Mul training/Adam/ReadVariableOp_140#training/Adam/mul_88/ReadVariableOp*
_output_shapes
:+*
T0
d
 training/Adam/ReadVariableOp_141ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
[
training/Adam/sub_54/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
v
training/Adam/sub_54Subtraining/Adam/sub_54/x training/Adam/ReadVariableOp_141*
T0*
_output_shapes
: 
�
training/Adam/Square_17Square8training/Adam/gradients/dense_5/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:+
o
training/Adam/mul_89Multraining/Adam/sub_54training/Adam/Square_17*
T0*
_output_shapes
:+
n
training/Adam/add_53AddV2training/Adam/mul_88training/Adam/mul_89*
T0*
_output_shapes
:+
i
training/Adam/mul_90Multraining/Adam/multraining/Adam/add_52*
T0*
_output_shapes
:+
[
training/Adam/Const_37Const*
dtype0*
_output_shapes
: *
valueB
 *    
[
training/Adam/Const_38Const*
valueB
 *  �*
dtype0*
_output_shapes
: 
�
&training/Adam/clip_by_value_18/MinimumMinimumtraining/Adam/add_53training/Adam/Const_38*
T0*
_output_shapes
:+
�
training/Adam/clip_by_value_18Maximum&training/Adam/clip_by_value_18/Minimumtraining/Adam/Const_37*
T0*
_output_shapes
:+
b
training/Adam/Sqrt_18Sqrttraining/Adam/clip_by_value_18*
_output_shapes
:+*
T0
[
training/Adam/add_54/yConst*
dtype0*
_output_shapes
: *
valueB
 *���3
q
training/Adam/add_54AddV2training/Adam/Sqrt_18training/Adam/add_54/y*
T0*
_output_shapes
:+
t
training/Adam/truediv_18RealDivtraining/Adam/mul_90training/Adam/add_54*
T0*
_output_shapes
:+
i
 training/Adam/ReadVariableOp_142ReadVariableOpdense_5/bias*
dtype0*
_output_shapes
:+
|
training/Adam/sub_55Sub training/Adam/ReadVariableOp_142training/Adam/truediv_18*
T0*
_output_shapes
:+
n
!training/Adam/AssignVariableOp_51AssignVariableOptraining/Adam/m_17_1training/Adam/add_52*
dtype0
�
 training/Adam/ReadVariableOp_143ReadVariableOptraining/Adam/m_17_1"^training/Adam/AssignVariableOp_51*
dtype0*
_output_shapes
:+
n
!training/Adam/AssignVariableOp_52AssignVariableOptraining/Adam/v_17_1training/Adam/add_53*
dtype0
�
 training/Adam/ReadVariableOp_144ReadVariableOptraining/Adam/v_17_1"^training/Adam/AssignVariableOp_52*
dtype0*
_output_shapes
:+
f
!training/Adam/AssignVariableOp_53AssignVariableOpdense_5/biastraining/Adam/sub_55*
dtype0
�
 training/Adam/ReadVariableOp_145ReadVariableOpdense_5/bias"^training/Adam/AssignVariableOp_53*
dtype0*
_output_shapes
:+
W
training/VarIsInitializedOpVarIsInitializedOpdense_1/bias*
_output_shapes
: 
R
training/VarIsInitializedOp_1VarIsInitializedOpcount*
_output_shapes
: 
`
training/VarIsInitializedOp_2VarIsInitializedOptraining/Adam/v_5_1*
_output_shapes
: 
a
training/VarIsInitializedOp_3VarIsInitializedOptraining/Adam/m_15_1*
_output_shapes
: 
d
training/VarIsInitializedOp_4VarIsInitializedOptraining/Adam/vhat_12_1*
_output_shapes
: 
a
training/VarIsInitializedOp_5VarIsInitializedOptraining/Adam/m_14_1*
_output_shapes
: 
n
training/VarIsInitializedOp_6VarIsInitializedOp!batch_normalization_3/moving_mean*
_output_shapes
: 
`
training/VarIsInitializedOp_7VarIsInitializedOptraining/Adam/m_3_1*
_output_shapes
: 
`
training/VarIsInitializedOp_8VarIsInitializedOptraining/Adam/m_8_1*
_output_shapes
: 
a
training/VarIsInitializedOp_9VarIsInitializedOptraining/Adam/m_16_1*
_output_shapes
: 
d
training/VarIsInitializedOp_10VarIsInitializedOptraining/Adam/vhat_5_1*
_output_shapes
: 
b
training/VarIsInitializedOp_11VarIsInitializedOptraining/Adam/v_15_1*
_output_shapes
: 
i
training/VarIsInitializedOp_12VarIsInitializedOpbatch_normalization_1/gamma*
_output_shapes
: 
o
training/VarIsInitializedOp_13VarIsInitializedOp!batch_normalization_2/moving_mean*
_output_shapes
: 
h
training/VarIsInitializedOp_14VarIsInitializedOpbatch_normalization_3/beta*
_output_shapes
: 
Z
training/VarIsInitializedOp_15VarIsInitializedOpdense_4/bias*
_output_shapes
: 
h
training/VarIsInitializedOp_16VarIsInitializedOpbatch_normalization_4/beta*
_output_shapes
: 
Y
training/VarIsInitializedOp_17VarIsInitializedOpAdam/beta_2*
_output_shapes
: 
e
training/VarIsInitializedOp_18VarIsInitializedOptraining/Adam/vhat_11_1*
_output_shapes
: 
e
training/VarIsInitializedOp_19VarIsInitializedOptraining/Adam/vhat_17_1*
_output_shapes
: 
b
training/VarIsInitializedOp_20VarIsInitializedOptraining/Adam/m_10_1*
_output_shapes
: 
a
training/VarIsInitializedOp_21VarIsInitializedOptraining/Adam/v_8_1*
_output_shapes
: 
b
training/VarIsInitializedOp_22VarIsInitializedOptraining/Adam/v_16_1*
_output_shapes
: 
b
training/VarIsInitializedOp_23VarIsInitializedOptraining/Adam/v_14_1*
_output_shapes
: 
d
training/VarIsInitializedOp_24VarIsInitializedOptraining/Adam/vhat_2_1*
_output_shapes
: 
s
training/VarIsInitializedOp_25VarIsInitializedOp%batch_normalization_1/moving_variance*
_output_shapes
: 
`
training/VarIsInitializedOp_26VarIsInitializedOpAdam/learning_rate*
_output_shapes
: 
a
training/VarIsInitializedOp_27VarIsInitializedOptraining/Adam/m_0_1*
_output_shapes
: 
d
training/VarIsInitializedOp_28VarIsInitializedOptraining/Adam/vhat_8_1*
_output_shapes
: 
e
training/VarIsInitializedOp_29VarIsInitializedOptraining/Adam/vhat_14_1*
_output_shapes
: 
i
training/VarIsInitializedOp_30VarIsInitializedOpbatch_normalization_2/gamma*
_output_shapes
: 
a
training/VarIsInitializedOp_31VarIsInitializedOptraining/Adam/m_4_1*
_output_shapes
: 
a
training/VarIsInitializedOp_32VarIsInitializedOptraining/Adam/m_9_1*
_output_shapes
: 
b
training/VarIsInitializedOp_33VarIsInitializedOptraining/Adam/m_13_1*
_output_shapes
: 
a
training/VarIsInitializedOp_34VarIsInitializedOptraining/Adam/v_4_1*
_output_shapes
: 
s
training/VarIsInitializedOp_35VarIsInitializedOp%batch_normalization_3/moving_variance*
_output_shapes
: 
a
training/VarIsInitializedOp_36VarIsInitializedOptraining/Adam/m_2_1*
_output_shapes
: 
a
training/VarIsInitializedOp_37VarIsInitializedOptraining/Adam/v_0_1*
_output_shapes
: 
b
training/VarIsInitializedOp_38VarIsInitializedOptraining/Adam/v_10_1*
_output_shapes
: 
\
training/VarIsInitializedOp_39VarIsInitializedOpdense_1/kernel*
_output_shapes
: 
\
training/VarIsInitializedOp_40VarIsInitializedOpdense_2/kernel*
_output_shapes
: 
a
training/VarIsInitializedOp_41VarIsInitializedOptraining/Adam/m_1_1*
_output_shapes
: 
a
training/VarIsInitializedOp_42VarIsInitializedOptraining/Adam/v_9_1*
_output_shapes
: 
a
training/VarIsInitializedOp_43VarIsInitializedOptraining/Adam/v_2_1*
_output_shapes
: 
b
training/VarIsInitializedOp_44VarIsInitializedOptraining/Adam/v_12_1*
_output_shapes
: 
d
training/VarIsInitializedOp_45VarIsInitializedOptraining/Adam/vhat_0_1*
_output_shapes
: 
d
training/VarIsInitializedOp_46VarIsInitializedOptraining/Adam/vhat_6_1*
_output_shapes
: 
b
training/VarIsInitializedOp_47VarIsInitializedOptraining/Adam/v_13_1*
_output_shapes
: 
i
training/VarIsInitializedOp_48VarIsInitializedOpbatch_normalization_3/gamma*
_output_shapes
: 
h
training/VarIsInitializedOp_49VarIsInitializedOpbatch_normalization_1/beta*
_output_shapes
: 
]
training/VarIsInitializedOp_50VarIsInitializedOpAdam/iterations*
_output_shapes
: 
S
training/VarIsInitializedOp_51VarIsInitializedOptotal*
_output_shapes
: 
b
training/VarIsInitializedOp_52VarIsInitializedOptraining/Adam/m_17_1*
_output_shapes
: 
d
training/VarIsInitializedOp_53VarIsInitializedOptraining/Adam/vhat_3_1*
_output_shapes
: 
h
training/VarIsInitializedOp_54VarIsInitializedOpbatch_normalization_2/beta*
_output_shapes
: 
s
training/VarIsInitializedOp_55VarIsInitializedOp%batch_normalization_4/moving_variance*
_output_shapes
: 
Z
training/VarIsInitializedOp_56VarIsInitializedOpdense_5/bias*
_output_shapes
: 
a
training/VarIsInitializedOp_57VarIsInitializedOptraining/Adam/v_1_1*
_output_shapes
: 
d
training/VarIsInitializedOp_58VarIsInitializedOptraining/Adam/vhat_9_1*
_output_shapes
: 
e
training/VarIsInitializedOp_59VarIsInitializedOptraining/Adam/vhat_15_1*
_output_shapes
: 
a
training/VarIsInitializedOp_60VarIsInitializedOptraining/Adam/m_7_1*
_output_shapes
: 
\
training/VarIsInitializedOp_61VarIsInitializedOpdense_4/kernel*
_output_shapes
: 
Y
training/VarIsInitializedOp_62VarIsInitializedOpAdam/beta_1*
_output_shapes
: 
b
training/VarIsInitializedOp_63VarIsInitializedOptraining/Adam/v_17_1*
_output_shapes
: 
a
training/VarIsInitializedOp_64VarIsInitializedOptraining/Adam/v_7_1*
_output_shapes
: 
o
training/VarIsInitializedOp_65VarIsInitializedOp!batch_normalization_4/moving_mean*
_output_shapes
: 
\
training/VarIsInitializedOp_66VarIsInitializedOpdense_3/kernel*
_output_shapes
: 
Z
training/VarIsInitializedOp_67VarIsInitializedOpdense_3/bias*
_output_shapes
: 
b
training/VarIsInitializedOp_68VarIsInitializedOptraining/Adam/m_11_1*
_output_shapes
: 
a
training/VarIsInitializedOp_69VarIsInitializedOptraining/Adam/v_3_1*
_output_shapes
: 
o
training/VarIsInitializedOp_70VarIsInitializedOp!batch_normalization_1/moving_mean*
_output_shapes
: 
i
training/VarIsInitializedOp_71VarIsInitializedOpbatch_normalization_4/gamma*
_output_shapes
: 
X
training/VarIsInitializedOp_72VarIsInitializedOp
Adam/decay*
_output_shapes
: 
a
training/VarIsInitializedOp_73VarIsInitializedOptraining/Adam/m_6_1*
_output_shapes
: 
d
training/VarIsInitializedOp_74VarIsInitializedOptraining/Adam/vhat_4_1*
_output_shapes
: 
b
training/VarIsInitializedOp_75VarIsInitializedOptraining/Adam/m_12_1*
_output_shapes
: 
e
training/VarIsInitializedOp_76VarIsInitializedOptraining/Adam/vhat_10_1*
_output_shapes
: 
e
training/VarIsInitializedOp_77VarIsInitializedOptraining/Adam/vhat_16_1*
_output_shapes
: 
a
training/VarIsInitializedOp_78VarIsInitializedOptraining/Adam/m_5_1*
_output_shapes
: 
Z
training/VarIsInitializedOp_79VarIsInitializedOpdense_2/bias*
_output_shapes
: 
d
training/VarIsInitializedOp_80VarIsInitializedOptraining/Adam/vhat_7_1*
_output_shapes
: 
d
training/VarIsInitializedOp_81VarIsInitializedOptraining/Adam/vhat_1_1*
_output_shapes
: 
b
training/VarIsInitializedOp_82VarIsInitializedOptraining/Adam/v_11_1*
_output_shapes
: 
e
training/VarIsInitializedOp_83VarIsInitializedOptraining/Adam/vhat_13_1*
_output_shapes
: 
a
training/VarIsInitializedOp_84VarIsInitializedOptraining/Adam/v_6_1*
_output_shapes
: 
s
training/VarIsInitializedOp_85VarIsInitializedOp%batch_normalization_2/moving_variance*
_output_shapes
: 
\
training/VarIsInitializedOp_86VarIsInitializedOpdense_5/kernel*
_output_shapes
: 
�
training/initNoOp^Adam/beta_1/Assign^Adam/beta_2/Assign^Adam/decay/Assign^Adam/iterations/Assign^Adam/learning_rate/Assign"^batch_normalization_1/beta/Assign#^batch_normalization_1/gamma/Assign)^batch_normalization_1/moving_mean/Assign-^batch_normalization_1/moving_variance/Assign"^batch_normalization_2/beta/Assign#^batch_normalization_2/gamma/Assign)^batch_normalization_2/moving_mean/Assign-^batch_normalization_2/moving_variance/Assign"^batch_normalization_3/beta/Assign#^batch_normalization_3/gamma/Assign)^batch_normalization_3/moving_mean/Assign-^batch_normalization_3/moving_variance/Assign"^batch_normalization_4/beta/Assign#^batch_normalization_4/gamma/Assign)^batch_normalization_4/moving_mean/Assign-^batch_normalization_4/moving_variance/Assign^count/Assign^dense_1/bias/Assign^dense_1/kernel/Assign^dense_2/bias/Assign^dense_2/kernel/Assign^dense_3/bias/Assign^dense_3/kernel/Assign^dense_4/bias/Assign^dense_4/kernel/Assign^dense_5/bias/Assign^dense_5/kernel/Assign^total/Assign^training/Adam/m_0_1/Assign^training/Adam/m_10_1/Assign^training/Adam/m_11_1/Assign^training/Adam/m_12_1/Assign^training/Adam/m_13_1/Assign^training/Adam/m_14_1/Assign^training/Adam/m_15_1/Assign^training/Adam/m_16_1/Assign^training/Adam/m_17_1/Assign^training/Adam/m_1_1/Assign^training/Adam/m_2_1/Assign^training/Adam/m_3_1/Assign^training/Adam/m_4_1/Assign^training/Adam/m_5_1/Assign^training/Adam/m_6_1/Assign^training/Adam/m_7_1/Assign^training/Adam/m_8_1/Assign^training/Adam/m_9_1/Assign^training/Adam/v_0_1/Assign^training/Adam/v_10_1/Assign^training/Adam/v_11_1/Assign^training/Adam/v_12_1/Assign^training/Adam/v_13_1/Assign^training/Adam/v_14_1/Assign^training/Adam/v_15_1/Assign^training/Adam/v_16_1/Assign^training/Adam/v_17_1/Assign^training/Adam/v_1_1/Assign^training/Adam/v_2_1/Assign^training/Adam/v_3_1/Assign^training/Adam/v_4_1/Assign^training/Adam/v_5_1/Assign^training/Adam/v_6_1/Assign^training/Adam/v_7_1/Assign^training/Adam/v_8_1/Assign^training/Adam/v_9_1/Assign^training/Adam/vhat_0_1/Assign^training/Adam/vhat_10_1/Assign^training/Adam/vhat_11_1/Assign^training/Adam/vhat_12_1/Assign^training/Adam/vhat_13_1/Assign^training/Adam/vhat_14_1/Assign^training/Adam/vhat_15_1/Assign^training/Adam/vhat_16_1/Assign^training/Adam/vhat_17_1/Assign^training/Adam/vhat_1_1/Assign^training/Adam/vhat_2_1/Assign^training/Adam/vhat_3_1/Assign^training/Adam/vhat_4_1/Assign^training/Adam/vhat_5_1/Assign^training/Adam/vhat_6_1/Assign^training/Adam/vhat_7_1/Assign^training/Adam/vhat_8_1/Assign^training/Adam/vhat_9_1/Assign
�
training/group_depsNoOp^Mean*^batch_normalization_1/AssignSubVariableOp,^batch_normalization_1/AssignSubVariableOp_1*^batch_normalization_2/AssignSubVariableOp,^batch_normalization_2/AssignSubVariableOp_1*^batch_normalization_3/AssignSubVariableOp,^batch_normalization_3/AssignSubVariableOp_1*^batch_normalization_4/AssignSubVariableOp,^batch_normalization_4/AssignSubVariableOp_1%^metrics/accuracy/AssignAddVariableOp'^metrics/accuracy/AssignAddVariableOp_1"^training/Adam/AssignAddVariableOp^training/Adam/AssignVariableOp!^training/Adam/AssignVariableOp_1"^training/Adam/AssignVariableOp_10"^training/Adam/AssignVariableOp_11"^training/Adam/AssignVariableOp_12"^training/Adam/AssignVariableOp_13"^training/Adam/AssignVariableOp_14"^training/Adam/AssignVariableOp_15"^training/Adam/AssignVariableOp_16"^training/Adam/AssignVariableOp_17"^training/Adam/AssignVariableOp_18"^training/Adam/AssignVariableOp_19!^training/Adam/AssignVariableOp_2"^training/Adam/AssignVariableOp_20"^training/Adam/AssignVariableOp_21"^training/Adam/AssignVariableOp_22"^training/Adam/AssignVariableOp_23"^training/Adam/AssignVariableOp_24"^training/Adam/AssignVariableOp_25"^training/Adam/AssignVariableOp_26"^training/Adam/AssignVariableOp_27"^training/Adam/AssignVariableOp_28"^training/Adam/AssignVariableOp_29!^training/Adam/AssignVariableOp_3"^training/Adam/AssignVariableOp_30"^training/Adam/AssignVariableOp_31"^training/Adam/AssignVariableOp_32"^training/Adam/AssignVariableOp_33"^training/Adam/AssignVariableOp_34"^training/Adam/AssignVariableOp_35"^training/Adam/AssignVariableOp_36"^training/Adam/AssignVariableOp_37"^training/Adam/AssignVariableOp_38"^training/Adam/AssignVariableOp_39!^training/Adam/AssignVariableOp_4"^training/Adam/AssignVariableOp_40"^training/Adam/AssignVariableOp_41"^training/Adam/AssignVariableOp_42"^training/Adam/AssignVariableOp_43"^training/Adam/AssignVariableOp_44"^training/Adam/AssignVariableOp_45"^training/Adam/AssignVariableOp_46"^training/Adam/AssignVariableOp_47"^training/Adam/AssignVariableOp_48"^training/Adam/AssignVariableOp_49!^training/Adam/AssignVariableOp_5"^training/Adam/AssignVariableOp_50"^training/Adam/AssignVariableOp_51"^training/Adam/AssignVariableOp_52"^training/Adam/AssignVariableOp_53!^training/Adam/AssignVariableOp_6!^training/Adam/AssignVariableOp_7!^training/Adam/AssignVariableOp_8!^training/Adam/AssignVariableOp_9
i

group_depsNoOp^Mean%^metrics/accuracy/AssignAddVariableOp'^metrics/accuracy/AssignAddVariableOp_1"�"�X
trainable_variables�X�X
m
dense_1/kernel:0dense_1/kernel/Assign$dense_1/kernel/Read/ReadVariableOp:0(2dense_1/random_uniform:08
^
dense_1/bias:0dense_1/bias/Assign"dense_1/bias/Read/ReadVariableOp:0(2dense_1/Const:08
�
batch_normalization_1/gamma:0"batch_normalization_1/gamma/Assign1batch_normalization_1/gamma/Read/ReadVariableOp:0(2batch_normalization_1/Const:08
�
batch_normalization_1/beta:0!batch_normalization_1/beta/Assign0batch_normalization_1/beta/Read/ReadVariableOp:0(2batch_normalization_1/Const_1:08
�
#batch_normalization_1/moving_mean:0(batch_normalization_1/moving_mean/Assign7batch_normalization_1/moving_mean/Read/ReadVariableOp:0(2batch_normalization_1/Const_2:08
�
'batch_normalization_1/moving_variance:0,batch_normalization_1/moving_variance/Assign;batch_normalization_1/moving_variance/Read/ReadVariableOp:0(2batch_normalization_1/Const_3:08
m
dense_2/kernel:0dense_2/kernel/Assign$dense_2/kernel/Read/ReadVariableOp:0(2dense_2/random_uniform:08
^
dense_2/bias:0dense_2/bias/Assign"dense_2/bias/Read/ReadVariableOp:0(2dense_2/Const:08
�
batch_normalization_2/gamma:0"batch_normalization_2/gamma/Assign1batch_normalization_2/gamma/Read/ReadVariableOp:0(2batch_normalization_2/Const:08
�
batch_normalization_2/beta:0!batch_normalization_2/beta/Assign0batch_normalization_2/beta/Read/ReadVariableOp:0(2batch_normalization_2/Const_1:08
�
#batch_normalization_2/moving_mean:0(batch_normalization_2/moving_mean/Assign7batch_normalization_2/moving_mean/Read/ReadVariableOp:0(2batch_normalization_2/Const_2:08
�
'batch_normalization_2/moving_variance:0,batch_normalization_2/moving_variance/Assign;batch_normalization_2/moving_variance/Read/ReadVariableOp:0(2batch_normalization_2/Const_3:08
m
dense_3/kernel:0dense_3/kernel/Assign$dense_3/kernel/Read/ReadVariableOp:0(2dense_3/random_uniform:08
^
dense_3/bias:0dense_3/bias/Assign"dense_3/bias/Read/ReadVariableOp:0(2dense_3/Const:08
�
batch_normalization_3/gamma:0"batch_normalization_3/gamma/Assign1batch_normalization_3/gamma/Read/ReadVariableOp:0(2batch_normalization_3/Const:08
�
batch_normalization_3/beta:0!batch_normalization_3/beta/Assign0batch_normalization_3/beta/Read/ReadVariableOp:0(2batch_normalization_3/Const_1:08
�
#batch_normalization_3/moving_mean:0(batch_normalization_3/moving_mean/Assign7batch_normalization_3/moving_mean/Read/ReadVariableOp:0(2batch_normalization_3/Const_2:08
�
'batch_normalization_3/moving_variance:0,batch_normalization_3/moving_variance/Assign;batch_normalization_3/moving_variance/Read/ReadVariableOp:0(2batch_normalization_3/Const_3:08
m
dense_4/kernel:0dense_4/kernel/Assign$dense_4/kernel/Read/ReadVariableOp:0(2dense_4/random_uniform:08
^
dense_4/bias:0dense_4/bias/Assign"dense_4/bias/Read/ReadVariableOp:0(2dense_4/Const:08
�
batch_normalization_4/gamma:0"batch_normalization_4/gamma/Assign1batch_normalization_4/gamma/Read/ReadVariableOp:0(2batch_normalization_4/Const:08
�
batch_normalization_4/beta:0!batch_normalization_4/beta/Assign0batch_normalization_4/beta/Read/ReadVariableOp:0(2batch_normalization_4/Const_1:08
�
#batch_normalization_4/moving_mean:0(batch_normalization_4/moving_mean/Assign7batch_normalization_4/moving_mean/Read/ReadVariableOp:0(2batch_normalization_4/Const_2:08
�
'batch_normalization_4/moving_variance:0,batch_normalization_4/moving_variance/Assign;batch_normalization_4/moving_variance/Read/ReadVariableOp:0(2batch_normalization_4/Const_3:08
m
dense_5/kernel:0dense_5/kernel/Assign$dense_5/kernel/Read/ReadVariableOp:0(2dense_5/random_uniform:08
^
dense_5/bias:0dense_5/bias/Assign"dense_5/bias/Read/ReadVariableOp:0(2dense_5/Const:08
�
Adam/iterations:0Adam/iterations/Assign%Adam/iterations/Read/ReadVariableOp:0(2+Adam/iterations/Initializer/initial_value:08
�
Adam/learning_rate:0Adam/learning_rate/Assign(Adam/learning_rate/Read/ReadVariableOp:0(2.Adam/learning_rate/Initializer/initial_value:08
s
Adam/beta_1:0Adam/beta_1/Assign!Adam/beta_1/Read/ReadVariableOp:0(2'Adam/beta_1/Initializer/initial_value:08
s
Adam/beta_2:0Adam/beta_2/Assign!Adam/beta_2/Read/ReadVariableOp:0(2'Adam/beta_2/Initializer/initial_value:08
o
Adam/decay:0Adam/decay/Assign Adam/decay/Read/ReadVariableOp:0(2&Adam/decay/Initializer/initial_value:08
A
total:0total/Assigntotal/Read/ReadVariableOp:0(2Const:08
C
count:0count/Assigncount/Read/ReadVariableOp:0(2	Const_1:08
w
training/Adam/m_0_1:0training/Adam/m_0_1/Assign)training/Adam/m_0_1/Read/ReadVariableOp:0(2training/Adam/m_0:08
w
training/Adam/m_1_1:0training/Adam/m_1_1/Assign)training/Adam/m_1_1/Read/ReadVariableOp:0(2training/Adam/m_1:08
w
training/Adam/m_2_1:0training/Adam/m_2_1/Assign)training/Adam/m_2_1/Read/ReadVariableOp:0(2training/Adam/m_2:08
w
training/Adam/m_3_1:0training/Adam/m_3_1/Assign)training/Adam/m_3_1/Read/ReadVariableOp:0(2training/Adam/m_3:08
w
training/Adam/m_4_1:0training/Adam/m_4_1/Assign)training/Adam/m_4_1/Read/ReadVariableOp:0(2training/Adam/m_4:08
w
training/Adam/m_5_1:0training/Adam/m_5_1/Assign)training/Adam/m_5_1/Read/ReadVariableOp:0(2training/Adam/m_5:08
w
training/Adam/m_6_1:0training/Adam/m_6_1/Assign)training/Adam/m_6_1/Read/ReadVariableOp:0(2training/Adam/m_6:08
w
training/Adam/m_7_1:0training/Adam/m_7_1/Assign)training/Adam/m_7_1/Read/ReadVariableOp:0(2training/Adam/m_7:08
w
training/Adam/m_8_1:0training/Adam/m_8_1/Assign)training/Adam/m_8_1/Read/ReadVariableOp:0(2training/Adam/m_8:08
w
training/Adam/m_9_1:0training/Adam/m_9_1/Assign)training/Adam/m_9_1/Read/ReadVariableOp:0(2training/Adam/m_9:08
{
training/Adam/m_10_1:0training/Adam/m_10_1/Assign*training/Adam/m_10_1/Read/ReadVariableOp:0(2training/Adam/m_10:08
{
training/Adam/m_11_1:0training/Adam/m_11_1/Assign*training/Adam/m_11_1/Read/ReadVariableOp:0(2training/Adam/m_11:08
{
training/Adam/m_12_1:0training/Adam/m_12_1/Assign*training/Adam/m_12_1/Read/ReadVariableOp:0(2training/Adam/m_12:08
{
training/Adam/m_13_1:0training/Adam/m_13_1/Assign*training/Adam/m_13_1/Read/ReadVariableOp:0(2training/Adam/m_13:08
{
training/Adam/m_14_1:0training/Adam/m_14_1/Assign*training/Adam/m_14_1/Read/ReadVariableOp:0(2training/Adam/m_14:08
{
training/Adam/m_15_1:0training/Adam/m_15_1/Assign*training/Adam/m_15_1/Read/ReadVariableOp:0(2training/Adam/m_15:08
{
training/Adam/m_16_1:0training/Adam/m_16_1/Assign*training/Adam/m_16_1/Read/ReadVariableOp:0(2training/Adam/m_16:08
{
training/Adam/m_17_1:0training/Adam/m_17_1/Assign*training/Adam/m_17_1/Read/ReadVariableOp:0(2training/Adam/m_17:08
w
training/Adam/v_0_1:0training/Adam/v_0_1/Assign)training/Adam/v_0_1/Read/ReadVariableOp:0(2training/Adam/v_0:08
w
training/Adam/v_1_1:0training/Adam/v_1_1/Assign)training/Adam/v_1_1/Read/ReadVariableOp:0(2training/Adam/v_1:08
w
training/Adam/v_2_1:0training/Adam/v_2_1/Assign)training/Adam/v_2_1/Read/ReadVariableOp:0(2training/Adam/v_2:08
w
training/Adam/v_3_1:0training/Adam/v_3_1/Assign)training/Adam/v_3_1/Read/ReadVariableOp:0(2training/Adam/v_3:08
w
training/Adam/v_4_1:0training/Adam/v_4_1/Assign)training/Adam/v_4_1/Read/ReadVariableOp:0(2training/Adam/v_4:08
w
training/Adam/v_5_1:0training/Adam/v_5_1/Assign)training/Adam/v_5_1/Read/ReadVariableOp:0(2training/Adam/v_5:08
w
training/Adam/v_6_1:0training/Adam/v_6_1/Assign)training/Adam/v_6_1/Read/ReadVariableOp:0(2training/Adam/v_6:08
w
training/Adam/v_7_1:0training/Adam/v_7_1/Assign)training/Adam/v_7_1/Read/ReadVariableOp:0(2training/Adam/v_7:08
w
training/Adam/v_8_1:0training/Adam/v_8_1/Assign)training/Adam/v_8_1/Read/ReadVariableOp:0(2training/Adam/v_8:08
w
training/Adam/v_9_1:0training/Adam/v_9_1/Assign)training/Adam/v_9_1/Read/ReadVariableOp:0(2training/Adam/v_9:08
{
training/Adam/v_10_1:0training/Adam/v_10_1/Assign*training/Adam/v_10_1/Read/ReadVariableOp:0(2training/Adam/v_10:08
{
training/Adam/v_11_1:0training/Adam/v_11_1/Assign*training/Adam/v_11_1/Read/ReadVariableOp:0(2training/Adam/v_11:08
{
training/Adam/v_12_1:0training/Adam/v_12_1/Assign*training/Adam/v_12_1/Read/ReadVariableOp:0(2training/Adam/v_12:08
{
training/Adam/v_13_1:0training/Adam/v_13_1/Assign*training/Adam/v_13_1/Read/ReadVariableOp:0(2training/Adam/v_13:08
{
training/Adam/v_14_1:0training/Adam/v_14_1/Assign*training/Adam/v_14_1/Read/ReadVariableOp:0(2training/Adam/v_14:08
{
training/Adam/v_15_1:0training/Adam/v_15_1/Assign*training/Adam/v_15_1/Read/ReadVariableOp:0(2training/Adam/v_15:08
{
training/Adam/v_16_1:0training/Adam/v_16_1/Assign*training/Adam/v_16_1/Read/ReadVariableOp:0(2training/Adam/v_16:08
{
training/Adam/v_17_1:0training/Adam/v_17_1/Assign*training/Adam/v_17_1/Read/ReadVariableOp:0(2training/Adam/v_17:08
�
training/Adam/vhat_0_1:0training/Adam/vhat_0_1/Assign,training/Adam/vhat_0_1/Read/ReadVariableOp:0(2training/Adam/vhat_0:08
�
training/Adam/vhat_1_1:0training/Adam/vhat_1_1/Assign,training/Adam/vhat_1_1/Read/ReadVariableOp:0(2training/Adam/vhat_1:08
�
training/Adam/vhat_2_1:0training/Adam/vhat_2_1/Assign,training/Adam/vhat_2_1/Read/ReadVariableOp:0(2training/Adam/vhat_2:08
�
training/Adam/vhat_3_1:0training/Adam/vhat_3_1/Assign,training/Adam/vhat_3_1/Read/ReadVariableOp:0(2training/Adam/vhat_3:08
�
training/Adam/vhat_4_1:0training/Adam/vhat_4_1/Assign,training/Adam/vhat_4_1/Read/ReadVariableOp:0(2training/Adam/vhat_4:08
�
training/Adam/vhat_5_1:0training/Adam/vhat_5_1/Assign,training/Adam/vhat_5_1/Read/ReadVariableOp:0(2training/Adam/vhat_5:08
�
training/Adam/vhat_6_1:0training/Adam/vhat_6_1/Assign,training/Adam/vhat_6_1/Read/ReadVariableOp:0(2training/Adam/vhat_6:08
�
training/Adam/vhat_7_1:0training/Adam/vhat_7_1/Assign,training/Adam/vhat_7_1/Read/ReadVariableOp:0(2training/Adam/vhat_7:08
�
training/Adam/vhat_8_1:0training/Adam/vhat_8_1/Assign,training/Adam/vhat_8_1/Read/ReadVariableOp:0(2training/Adam/vhat_8:08
�
training/Adam/vhat_9_1:0training/Adam/vhat_9_1/Assign,training/Adam/vhat_9_1/Read/ReadVariableOp:0(2training/Adam/vhat_9:08
�
training/Adam/vhat_10_1:0training/Adam/vhat_10_1/Assign-training/Adam/vhat_10_1/Read/ReadVariableOp:0(2training/Adam/vhat_10:08
�
training/Adam/vhat_11_1:0training/Adam/vhat_11_1/Assign-training/Adam/vhat_11_1/Read/ReadVariableOp:0(2training/Adam/vhat_11:08
�
training/Adam/vhat_12_1:0training/Adam/vhat_12_1/Assign-training/Adam/vhat_12_1/Read/ReadVariableOp:0(2training/Adam/vhat_12:08
�
training/Adam/vhat_13_1:0training/Adam/vhat_13_1/Assign-training/Adam/vhat_13_1/Read/ReadVariableOp:0(2training/Adam/vhat_13:08
�
training/Adam/vhat_14_1:0training/Adam/vhat_14_1/Assign-training/Adam/vhat_14_1/Read/ReadVariableOp:0(2training/Adam/vhat_14:08
�
training/Adam/vhat_15_1:0training/Adam/vhat_15_1/Assign-training/Adam/vhat_15_1/Read/ReadVariableOp:0(2training/Adam/vhat_15:08
�
training/Adam/vhat_16_1:0training/Adam/vhat_16_1/Assign-training/Adam/vhat_16_1/Read/ReadVariableOp:0(2training/Adam/vhat_16:08
�
training/Adam/vhat_17_1:0training/Adam/vhat_17_1/Assign-training/Adam/vhat_17_1/Read/ReadVariableOp:0(2training/Adam/vhat_17:08"�[
cond_context�[�[
�
$batch_normalization_1/cond/cond_text$batch_normalization_1/cond/pred_id:0%batch_normalization_1/cond/switch_t:0 *�
'batch_normalization_1/batchnorm/add_1:0
%batch_normalization_1/cond/Switch_1:0
%batch_normalization_1/cond/Switch_1:1
$batch_normalization_1/cond/pred_id:0
%batch_normalization_1/cond/switch_t:0P
'batch_normalization_1/batchnorm/add_1:0%batch_normalization_1/cond/Switch_1:1L
$batch_normalization_1/cond/pred_id:0$batch_normalization_1/cond/pred_id:0
�
&batch_normalization_1/cond/cond_text_1$batch_normalization_1/cond/pred_id:0%batch_normalization_1/cond/switch_f:0*�
batch_normalization_1/beta:0
<batch_normalization_1/cond/batchnorm/ReadVariableOp/Switch:0
5batch_normalization_1/cond/batchnorm/ReadVariableOp:0
>batch_normalization_1/cond/batchnorm/ReadVariableOp_1/Switch:0
7batch_normalization_1/cond/batchnorm/ReadVariableOp_1:0
>batch_normalization_1/cond/batchnorm/ReadVariableOp_2/Switch:0
7batch_normalization_1/cond/batchnorm/ReadVariableOp_2:0
,batch_normalization_1/cond/batchnorm/Rsqrt:0
,batch_normalization_1/cond/batchnorm/add/y:0
*batch_normalization_1/cond/batchnorm/add:0
,batch_normalization_1/cond/batchnorm/add_1:0
@batch_normalization_1/cond/batchnorm/mul/ReadVariableOp/Switch:0
9batch_normalization_1/cond/batchnorm/mul/ReadVariableOp:0
*batch_normalization_1/cond/batchnorm/mul:0
3batch_normalization_1/cond/batchnorm/mul_1/Switch:0
,batch_normalization_1/cond/batchnorm/mul_1:0
,batch_normalization_1/cond/batchnorm/mul_2:0
*batch_normalization_1/cond/batchnorm/sub:0
$batch_normalization_1/cond/pred_id:0
%batch_normalization_1/cond/switch_f:0
batch_normalization_1/gamma:0
#batch_normalization_1/moving_mean:0
'batch_normalization_1/moving_variance:0
dense_1/Relu:0a
batch_normalization_1/gamma:0@batch_normalization_1/cond/batchnorm/mul/ReadVariableOp/Switch:0E
dense_1/Relu:03batch_normalization_1/cond/batchnorm/mul_1/Switch:0L
$batch_normalization_1/cond/pred_id:0$batch_normalization_1/cond/pred_id:0g
'batch_normalization_1/moving_variance:0<batch_normalization_1/cond/batchnorm/ReadVariableOp/Switch:0e
#batch_normalization_1/moving_mean:0>batch_normalization_1/cond/batchnorm/ReadVariableOp_1/Switch:0^
batch_normalization_1/beta:0>batch_normalization_1/cond/batchnorm/ReadVariableOp_2/Switch:0
�
$batch_normalization_2/cond/cond_text$batch_normalization_2/cond/pred_id:0%batch_normalization_2/cond/switch_t:0 *�
'batch_normalization_2/batchnorm/add_1:0
%batch_normalization_2/cond/Switch_1:0
%batch_normalization_2/cond/Switch_1:1
$batch_normalization_2/cond/pred_id:0
%batch_normalization_2/cond/switch_t:0L
$batch_normalization_2/cond/pred_id:0$batch_normalization_2/cond/pred_id:0P
'batch_normalization_2/batchnorm/add_1:0%batch_normalization_2/cond/Switch_1:1
�
&batch_normalization_2/cond/cond_text_1$batch_normalization_2/cond/pred_id:0%batch_normalization_2/cond/switch_f:0*�
batch_normalization_2/beta:0
<batch_normalization_2/cond/batchnorm/ReadVariableOp/Switch:0
5batch_normalization_2/cond/batchnorm/ReadVariableOp:0
>batch_normalization_2/cond/batchnorm/ReadVariableOp_1/Switch:0
7batch_normalization_2/cond/batchnorm/ReadVariableOp_1:0
>batch_normalization_2/cond/batchnorm/ReadVariableOp_2/Switch:0
7batch_normalization_2/cond/batchnorm/ReadVariableOp_2:0
,batch_normalization_2/cond/batchnorm/Rsqrt:0
,batch_normalization_2/cond/batchnorm/add/y:0
*batch_normalization_2/cond/batchnorm/add:0
,batch_normalization_2/cond/batchnorm/add_1:0
@batch_normalization_2/cond/batchnorm/mul/ReadVariableOp/Switch:0
9batch_normalization_2/cond/batchnorm/mul/ReadVariableOp:0
*batch_normalization_2/cond/batchnorm/mul:0
3batch_normalization_2/cond/batchnorm/mul_1/Switch:0
,batch_normalization_2/cond/batchnorm/mul_1:0
,batch_normalization_2/cond/batchnorm/mul_2:0
*batch_normalization_2/cond/batchnorm/sub:0
$batch_normalization_2/cond/pred_id:0
%batch_normalization_2/cond/switch_f:0
batch_normalization_2/gamma:0
#batch_normalization_2/moving_mean:0
'batch_normalization_2/moving_variance:0
dense_2/Relu:0a
batch_normalization_2/gamma:0@batch_normalization_2/cond/batchnorm/mul/ReadVariableOp/Switch:0E
dense_2/Relu:03batch_normalization_2/cond/batchnorm/mul_1/Switch:0g
'batch_normalization_2/moving_variance:0<batch_normalization_2/cond/batchnorm/ReadVariableOp/Switch:0e
#batch_normalization_2/moving_mean:0>batch_normalization_2/cond/batchnorm/ReadVariableOp_1/Switch:0L
$batch_normalization_2/cond/pred_id:0$batch_normalization_2/cond/pred_id:0^
batch_normalization_2/beta:0>batch_normalization_2/cond/batchnorm/ReadVariableOp_2/Switch:0
�
dropout_1/cond/cond_textdropout_1/cond/pred_id:0dropout_1/cond/switch_t:0 *�
"batch_normalization_2/cond/Merge:0
dropout_1/cond/dropout/Cast:0
%dropout_1/cond/dropout/GreaterEqual:0
%dropout_1/cond/dropout/Shape/Switch:1
dropout_1/cond/dropout/Shape:0
dropout_1/cond/dropout/mul:0
dropout_1/cond/dropout/mul_1:0
5dropout_1/cond/dropout/random_uniform/RandomUniform:0
+dropout_1/cond/dropout/random_uniform/max:0
+dropout_1/cond/dropout/random_uniform/min:0
+dropout_1/cond/dropout/random_uniform/mul:0
+dropout_1/cond/dropout/random_uniform/sub:0
'dropout_1/cond/dropout/random_uniform:0
dropout_1/cond/dropout/rate:0
dropout_1/cond/dropout/sub/x:0
dropout_1/cond/dropout/sub:0
"dropout_1/cond/dropout/truediv/x:0
 dropout_1/cond/dropout/truediv:0
dropout_1/cond/pred_id:0
dropout_1/cond/switch_t:04
dropout_1/cond/pred_id:0dropout_1/cond/pred_id:0K
"batch_normalization_2/cond/Merge:0%dropout_1/cond/dropout/Shape/Switch:1
�
dropout_1/cond/cond_text_1dropout_1/cond/pred_id:0dropout_1/cond/switch_f:0*�
"batch_normalization_2/cond/Merge:0
dropout_1/cond/Switch_1:0
dropout_1/cond/Switch_1:1
dropout_1/cond/pred_id:0
dropout_1/cond/switch_f:04
dropout_1/cond/pred_id:0dropout_1/cond/pred_id:0?
"batch_normalization_2/cond/Merge:0dropout_1/cond/Switch_1:0
�
$batch_normalization_3/cond/cond_text$batch_normalization_3/cond/pred_id:0%batch_normalization_3/cond/switch_t:0 *�
'batch_normalization_3/batchnorm/add_1:0
%batch_normalization_3/cond/Switch_1:0
%batch_normalization_3/cond/Switch_1:1
$batch_normalization_3/cond/pred_id:0
%batch_normalization_3/cond/switch_t:0L
$batch_normalization_3/cond/pred_id:0$batch_normalization_3/cond/pred_id:0P
'batch_normalization_3/batchnorm/add_1:0%batch_normalization_3/cond/Switch_1:1
�
&batch_normalization_3/cond/cond_text_1$batch_normalization_3/cond/pred_id:0%batch_normalization_3/cond/switch_f:0*�
batch_normalization_3/beta:0
<batch_normalization_3/cond/batchnorm/ReadVariableOp/Switch:0
5batch_normalization_3/cond/batchnorm/ReadVariableOp:0
>batch_normalization_3/cond/batchnorm/ReadVariableOp_1/Switch:0
7batch_normalization_3/cond/batchnorm/ReadVariableOp_1:0
>batch_normalization_3/cond/batchnorm/ReadVariableOp_2/Switch:0
7batch_normalization_3/cond/batchnorm/ReadVariableOp_2:0
,batch_normalization_3/cond/batchnorm/Rsqrt:0
,batch_normalization_3/cond/batchnorm/add/y:0
*batch_normalization_3/cond/batchnorm/add:0
,batch_normalization_3/cond/batchnorm/add_1:0
@batch_normalization_3/cond/batchnorm/mul/ReadVariableOp/Switch:0
9batch_normalization_3/cond/batchnorm/mul/ReadVariableOp:0
*batch_normalization_3/cond/batchnorm/mul:0
3batch_normalization_3/cond/batchnorm/mul_1/Switch:0
,batch_normalization_3/cond/batchnorm/mul_1:0
,batch_normalization_3/cond/batchnorm/mul_2:0
*batch_normalization_3/cond/batchnorm/sub:0
$batch_normalization_3/cond/pred_id:0
%batch_normalization_3/cond/switch_f:0
batch_normalization_3/gamma:0
#batch_normalization_3/moving_mean:0
'batch_normalization_3/moving_variance:0
dense_3/Relu:0^
batch_normalization_3/beta:0>batch_normalization_3/cond/batchnorm/ReadVariableOp_2/Switch:0g
'batch_normalization_3/moving_variance:0<batch_normalization_3/cond/batchnorm/ReadVariableOp/Switch:0e
#batch_normalization_3/moving_mean:0>batch_normalization_3/cond/batchnorm/ReadVariableOp_1/Switch:0E
dense_3/Relu:03batch_normalization_3/cond/batchnorm/mul_1/Switch:0L
$batch_normalization_3/cond/pred_id:0$batch_normalization_3/cond/pred_id:0a
batch_normalization_3/gamma:0@batch_normalization_3/cond/batchnorm/mul/ReadVariableOp/Switch:0
�
dropout_2/cond/cond_textdropout_2/cond/pred_id:0dropout_2/cond/switch_t:0 *�
"batch_normalization_3/cond/Merge:0
dropout_2/cond/dropout/Cast:0
%dropout_2/cond/dropout/GreaterEqual:0
%dropout_2/cond/dropout/Shape/Switch:1
dropout_2/cond/dropout/Shape:0
dropout_2/cond/dropout/mul:0
dropout_2/cond/dropout/mul_1:0
5dropout_2/cond/dropout/random_uniform/RandomUniform:0
+dropout_2/cond/dropout/random_uniform/max:0
+dropout_2/cond/dropout/random_uniform/min:0
+dropout_2/cond/dropout/random_uniform/mul:0
+dropout_2/cond/dropout/random_uniform/sub:0
'dropout_2/cond/dropout/random_uniform:0
dropout_2/cond/dropout/rate:0
dropout_2/cond/dropout/sub/x:0
dropout_2/cond/dropout/sub:0
"dropout_2/cond/dropout/truediv/x:0
 dropout_2/cond/dropout/truediv:0
dropout_2/cond/pred_id:0
dropout_2/cond/switch_t:0K
"batch_normalization_3/cond/Merge:0%dropout_2/cond/dropout/Shape/Switch:14
dropout_2/cond/pred_id:0dropout_2/cond/pred_id:0
�
dropout_2/cond/cond_text_1dropout_2/cond/pred_id:0dropout_2/cond/switch_f:0*�
"batch_normalization_3/cond/Merge:0
dropout_2/cond/Switch_1:0
dropout_2/cond/Switch_1:1
dropout_2/cond/pred_id:0
dropout_2/cond/switch_f:0?
"batch_normalization_3/cond/Merge:0dropout_2/cond/Switch_1:04
dropout_2/cond/pred_id:0dropout_2/cond/pred_id:0
�
$batch_normalization_4/cond/cond_text$batch_normalization_4/cond/pred_id:0%batch_normalization_4/cond/switch_t:0 *�
'batch_normalization_4/batchnorm/add_1:0
%batch_normalization_4/cond/Switch_1:0
%batch_normalization_4/cond/Switch_1:1
$batch_normalization_4/cond/pred_id:0
%batch_normalization_4/cond/switch_t:0L
$batch_normalization_4/cond/pred_id:0$batch_normalization_4/cond/pred_id:0P
'batch_normalization_4/batchnorm/add_1:0%batch_normalization_4/cond/Switch_1:1
�
&batch_normalization_4/cond/cond_text_1$batch_normalization_4/cond/pred_id:0%batch_normalization_4/cond/switch_f:0*�
batch_normalization_4/beta:0
<batch_normalization_4/cond/batchnorm/ReadVariableOp/Switch:0
5batch_normalization_4/cond/batchnorm/ReadVariableOp:0
>batch_normalization_4/cond/batchnorm/ReadVariableOp_1/Switch:0
7batch_normalization_4/cond/batchnorm/ReadVariableOp_1:0
>batch_normalization_4/cond/batchnorm/ReadVariableOp_2/Switch:0
7batch_normalization_4/cond/batchnorm/ReadVariableOp_2:0
,batch_normalization_4/cond/batchnorm/Rsqrt:0
,batch_normalization_4/cond/batchnorm/add/y:0
*batch_normalization_4/cond/batchnorm/add:0
,batch_normalization_4/cond/batchnorm/add_1:0
@batch_normalization_4/cond/batchnorm/mul/ReadVariableOp/Switch:0
9batch_normalization_4/cond/batchnorm/mul/ReadVariableOp:0
*batch_normalization_4/cond/batchnorm/mul:0
3batch_normalization_4/cond/batchnorm/mul_1/Switch:0
,batch_normalization_4/cond/batchnorm/mul_1:0
,batch_normalization_4/cond/batchnorm/mul_2:0
*batch_normalization_4/cond/batchnorm/sub:0
$batch_normalization_4/cond/pred_id:0
%batch_normalization_4/cond/switch_f:0
batch_normalization_4/gamma:0
#batch_normalization_4/moving_mean:0
'batch_normalization_4/moving_variance:0
dense_4/Relu:0L
$batch_normalization_4/cond/pred_id:0$batch_normalization_4/cond/pred_id:0a
batch_normalization_4/gamma:0@batch_normalization_4/cond/batchnorm/mul/ReadVariableOp/Switch:0e
#batch_normalization_4/moving_mean:0>batch_normalization_4/cond/batchnorm/ReadVariableOp_1/Switch:0^
batch_normalization_4/beta:0>batch_normalization_4/cond/batchnorm/ReadVariableOp_2/Switch:0g
'batch_normalization_4/moving_variance:0<batch_normalization_4/cond/batchnorm/ReadVariableOp/Switch:0E
dense_4/Relu:03batch_normalization_4/cond/batchnorm/mul_1/Switch:0"�X
	variables�X�X
m
dense_1/kernel:0dense_1/kernel/Assign$dense_1/kernel/Read/ReadVariableOp:0(2dense_1/random_uniform:08
^
dense_1/bias:0dense_1/bias/Assign"dense_1/bias/Read/ReadVariableOp:0(2dense_1/Const:08
�
batch_normalization_1/gamma:0"batch_normalization_1/gamma/Assign1batch_normalization_1/gamma/Read/ReadVariableOp:0(2batch_normalization_1/Const:08
�
batch_normalization_1/beta:0!batch_normalization_1/beta/Assign0batch_normalization_1/beta/Read/ReadVariableOp:0(2batch_normalization_1/Const_1:08
�
#batch_normalization_1/moving_mean:0(batch_normalization_1/moving_mean/Assign7batch_normalization_1/moving_mean/Read/ReadVariableOp:0(2batch_normalization_1/Const_2:08
�
'batch_normalization_1/moving_variance:0,batch_normalization_1/moving_variance/Assign;batch_normalization_1/moving_variance/Read/ReadVariableOp:0(2batch_normalization_1/Const_3:08
m
dense_2/kernel:0dense_2/kernel/Assign$dense_2/kernel/Read/ReadVariableOp:0(2dense_2/random_uniform:08
^
dense_2/bias:0dense_2/bias/Assign"dense_2/bias/Read/ReadVariableOp:0(2dense_2/Const:08
�
batch_normalization_2/gamma:0"batch_normalization_2/gamma/Assign1batch_normalization_2/gamma/Read/ReadVariableOp:0(2batch_normalization_2/Const:08
�
batch_normalization_2/beta:0!batch_normalization_2/beta/Assign0batch_normalization_2/beta/Read/ReadVariableOp:0(2batch_normalization_2/Const_1:08
�
#batch_normalization_2/moving_mean:0(batch_normalization_2/moving_mean/Assign7batch_normalization_2/moving_mean/Read/ReadVariableOp:0(2batch_normalization_2/Const_2:08
�
'batch_normalization_2/moving_variance:0,batch_normalization_2/moving_variance/Assign;batch_normalization_2/moving_variance/Read/ReadVariableOp:0(2batch_normalization_2/Const_3:08
m
dense_3/kernel:0dense_3/kernel/Assign$dense_3/kernel/Read/ReadVariableOp:0(2dense_3/random_uniform:08
^
dense_3/bias:0dense_3/bias/Assign"dense_3/bias/Read/ReadVariableOp:0(2dense_3/Const:08
�
batch_normalization_3/gamma:0"batch_normalization_3/gamma/Assign1batch_normalization_3/gamma/Read/ReadVariableOp:0(2batch_normalization_3/Const:08
�
batch_normalization_3/beta:0!batch_normalization_3/beta/Assign0batch_normalization_3/beta/Read/ReadVariableOp:0(2batch_normalization_3/Const_1:08
�
#batch_normalization_3/moving_mean:0(batch_normalization_3/moving_mean/Assign7batch_normalization_3/moving_mean/Read/ReadVariableOp:0(2batch_normalization_3/Const_2:08
�
'batch_normalization_3/moving_variance:0,batch_normalization_3/moving_variance/Assign;batch_normalization_3/moving_variance/Read/ReadVariableOp:0(2batch_normalization_3/Const_3:08
m
dense_4/kernel:0dense_4/kernel/Assign$dense_4/kernel/Read/ReadVariableOp:0(2dense_4/random_uniform:08
^
dense_4/bias:0dense_4/bias/Assign"dense_4/bias/Read/ReadVariableOp:0(2dense_4/Const:08
�
batch_normalization_4/gamma:0"batch_normalization_4/gamma/Assign1batch_normalization_4/gamma/Read/ReadVariableOp:0(2batch_normalization_4/Const:08
�
batch_normalization_4/beta:0!batch_normalization_4/beta/Assign0batch_normalization_4/beta/Read/ReadVariableOp:0(2batch_normalization_4/Const_1:08
�
#batch_normalization_4/moving_mean:0(batch_normalization_4/moving_mean/Assign7batch_normalization_4/moving_mean/Read/ReadVariableOp:0(2batch_normalization_4/Const_2:08
�
'batch_normalization_4/moving_variance:0,batch_normalization_4/moving_variance/Assign;batch_normalization_4/moving_variance/Read/ReadVariableOp:0(2batch_normalization_4/Const_3:08
m
dense_5/kernel:0dense_5/kernel/Assign$dense_5/kernel/Read/ReadVariableOp:0(2dense_5/random_uniform:08
^
dense_5/bias:0dense_5/bias/Assign"dense_5/bias/Read/ReadVariableOp:0(2dense_5/Const:08
�
Adam/iterations:0Adam/iterations/Assign%Adam/iterations/Read/ReadVariableOp:0(2+Adam/iterations/Initializer/initial_value:08
�
Adam/learning_rate:0Adam/learning_rate/Assign(Adam/learning_rate/Read/ReadVariableOp:0(2.Adam/learning_rate/Initializer/initial_value:08
s
Adam/beta_1:0Adam/beta_1/Assign!Adam/beta_1/Read/ReadVariableOp:0(2'Adam/beta_1/Initializer/initial_value:08
s
Adam/beta_2:0Adam/beta_2/Assign!Adam/beta_2/Read/ReadVariableOp:0(2'Adam/beta_2/Initializer/initial_value:08
o
Adam/decay:0Adam/decay/Assign Adam/decay/Read/ReadVariableOp:0(2&Adam/decay/Initializer/initial_value:08
A
total:0total/Assigntotal/Read/ReadVariableOp:0(2Const:08
C
count:0count/Assigncount/Read/ReadVariableOp:0(2	Const_1:08
w
training/Adam/m_0_1:0training/Adam/m_0_1/Assign)training/Adam/m_0_1/Read/ReadVariableOp:0(2training/Adam/m_0:08
w
training/Adam/m_1_1:0training/Adam/m_1_1/Assign)training/Adam/m_1_1/Read/ReadVariableOp:0(2training/Adam/m_1:08
w
training/Adam/m_2_1:0training/Adam/m_2_1/Assign)training/Adam/m_2_1/Read/ReadVariableOp:0(2training/Adam/m_2:08
w
training/Adam/m_3_1:0training/Adam/m_3_1/Assign)training/Adam/m_3_1/Read/ReadVariableOp:0(2training/Adam/m_3:08
w
training/Adam/m_4_1:0training/Adam/m_4_1/Assign)training/Adam/m_4_1/Read/ReadVariableOp:0(2training/Adam/m_4:08
w
training/Adam/m_5_1:0training/Adam/m_5_1/Assign)training/Adam/m_5_1/Read/ReadVariableOp:0(2training/Adam/m_5:08
w
training/Adam/m_6_1:0training/Adam/m_6_1/Assign)training/Adam/m_6_1/Read/ReadVariableOp:0(2training/Adam/m_6:08
w
training/Adam/m_7_1:0training/Adam/m_7_1/Assign)training/Adam/m_7_1/Read/ReadVariableOp:0(2training/Adam/m_7:08
w
training/Adam/m_8_1:0training/Adam/m_8_1/Assign)training/Adam/m_8_1/Read/ReadVariableOp:0(2training/Adam/m_8:08
w
training/Adam/m_9_1:0training/Adam/m_9_1/Assign)training/Adam/m_9_1/Read/ReadVariableOp:0(2training/Adam/m_9:08
{
training/Adam/m_10_1:0training/Adam/m_10_1/Assign*training/Adam/m_10_1/Read/ReadVariableOp:0(2training/Adam/m_10:08
{
training/Adam/m_11_1:0training/Adam/m_11_1/Assign*training/Adam/m_11_1/Read/ReadVariableOp:0(2training/Adam/m_11:08
{
training/Adam/m_12_1:0training/Adam/m_12_1/Assign*training/Adam/m_12_1/Read/ReadVariableOp:0(2training/Adam/m_12:08
{
training/Adam/m_13_1:0training/Adam/m_13_1/Assign*training/Adam/m_13_1/Read/ReadVariableOp:0(2training/Adam/m_13:08
{
training/Adam/m_14_1:0training/Adam/m_14_1/Assign*training/Adam/m_14_1/Read/ReadVariableOp:0(2training/Adam/m_14:08
{
training/Adam/m_15_1:0training/Adam/m_15_1/Assign*training/Adam/m_15_1/Read/ReadVariableOp:0(2training/Adam/m_15:08
{
training/Adam/m_16_1:0training/Adam/m_16_1/Assign*training/Adam/m_16_1/Read/ReadVariableOp:0(2training/Adam/m_16:08
{
training/Adam/m_17_1:0training/Adam/m_17_1/Assign*training/Adam/m_17_1/Read/ReadVariableOp:0(2training/Adam/m_17:08
w
training/Adam/v_0_1:0training/Adam/v_0_1/Assign)training/Adam/v_0_1/Read/ReadVariableOp:0(2training/Adam/v_0:08
w
training/Adam/v_1_1:0training/Adam/v_1_1/Assign)training/Adam/v_1_1/Read/ReadVariableOp:0(2training/Adam/v_1:08
w
training/Adam/v_2_1:0training/Adam/v_2_1/Assign)training/Adam/v_2_1/Read/ReadVariableOp:0(2training/Adam/v_2:08
w
training/Adam/v_3_1:0training/Adam/v_3_1/Assign)training/Adam/v_3_1/Read/ReadVariableOp:0(2training/Adam/v_3:08
w
training/Adam/v_4_1:0training/Adam/v_4_1/Assign)training/Adam/v_4_1/Read/ReadVariableOp:0(2training/Adam/v_4:08
w
training/Adam/v_5_1:0training/Adam/v_5_1/Assign)training/Adam/v_5_1/Read/ReadVariableOp:0(2training/Adam/v_5:08
w
training/Adam/v_6_1:0training/Adam/v_6_1/Assign)training/Adam/v_6_1/Read/ReadVariableOp:0(2training/Adam/v_6:08
w
training/Adam/v_7_1:0training/Adam/v_7_1/Assign)training/Adam/v_7_1/Read/ReadVariableOp:0(2training/Adam/v_7:08
w
training/Adam/v_8_1:0training/Adam/v_8_1/Assign)training/Adam/v_8_1/Read/ReadVariableOp:0(2training/Adam/v_8:08
w
training/Adam/v_9_1:0training/Adam/v_9_1/Assign)training/Adam/v_9_1/Read/ReadVariableOp:0(2training/Adam/v_9:08
{
training/Adam/v_10_1:0training/Adam/v_10_1/Assign*training/Adam/v_10_1/Read/ReadVariableOp:0(2training/Adam/v_10:08
{
training/Adam/v_11_1:0training/Adam/v_11_1/Assign*training/Adam/v_11_1/Read/ReadVariableOp:0(2training/Adam/v_11:08
{
training/Adam/v_12_1:0training/Adam/v_12_1/Assign*training/Adam/v_12_1/Read/ReadVariableOp:0(2training/Adam/v_12:08
{
training/Adam/v_13_1:0training/Adam/v_13_1/Assign*training/Adam/v_13_1/Read/ReadVariableOp:0(2training/Adam/v_13:08
{
training/Adam/v_14_1:0training/Adam/v_14_1/Assign*training/Adam/v_14_1/Read/ReadVariableOp:0(2training/Adam/v_14:08
{
training/Adam/v_15_1:0training/Adam/v_15_1/Assign*training/Adam/v_15_1/Read/ReadVariableOp:0(2training/Adam/v_15:08
{
training/Adam/v_16_1:0training/Adam/v_16_1/Assign*training/Adam/v_16_1/Read/ReadVariableOp:0(2training/Adam/v_16:08
{
training/Adam/v_17_1:0training/Adam/v_17_1/Assign*training/Adam/v_17_1/Read/ReadVariableOp:0(2training/Adam/v_17:08
�
training/Adam/vhat_0_1:0training/Adam/vhat_0_1/Assign,training/Adam/vhat_0_1/Read/ReadVariableOp:0(2training/Adam/vhat_0:08
�
training/Adam/vhat_1_1:0training/Adam/vhat_1_1/Assign,training/Adam/vhat_1_1/Read/ReadVariableOp:0(2training/Adam/vhat_1:08
�
training/Adam/vhat_2_1:0training/Adam/vhat_2_1/Assign,training/Adam/vhat_2_1/Read/ReadVariableOp:0(2training/Adam/vhat_2:08
�
training/Adam/vhat_3_1:0training/Adam/vhat_3_1/Assign,training/Adam/vhat_3_1/Read/ReadVariableOp:0(2training/Adam/vhat_3:08
�
training/Adam/vhat_4_1:0training/Adam/vhat_4_1/Assign,training/Adam/vhat_4_1/Read/ReadVariableOp:0(2training/Adam/vhat_4:08
�
training/Adam/vhat_5_1:0training/Adam/vhat_5_1/Assign,training/Adam/vhat_5_1/Read/ReadVariableOp:0(2training/Adam/vhat_5:08
�
training/Adam/vhat_6_1:0training/Adam/vhat_6_1/Assign,training/Adam/vhat_6_1/Read/ReadVariableOp:0(2training/Adam/vhat_6:08
�
training/Adam/vhat_7_1:0training/Adam/vhat_7_1/Assign,training/Adam/vhat_7_1/Read/ReadVariableOp:0(2training/Adam/vhat_7:08
�
training/Adam/vhat_8_1:0training/Adam/vhat_8_1/Assign,training/Adam/vhat_8_1/Read/ReadVariableOp:0(2training/Adam/vhat_8:08
�
training/Adam/vhat_9_1:0training/Adam/vhat_9_1/Assign,training/Adam/vhat_9_1/Read/ReadVariableOp:0(2training/Adam/vhat_9:08
�
training/Adam/vhat_10_1:0training/Adam/vhat_10_1/Assign-training/Adam/vhat_10_1/Read/ReadVariableOp:0(2training/Adam/vhat_10:08
�
training/Adam/vhat_11_1:0training/Adam/vhat_11_1/Assign-training/Adam/vhat_11_1/Read/ReadVariableOp:0(2training/Adam/vhat_11:08
�
training/Adam/vhat_12_1:0training/Adam/vhat_12_1/Assign-training/Adam/vhat_12_1/Read/ReadVariableOp:0(2training/Adam/vhat_12:08
�
training/Adam/vhat_13_1:0training/Adam/vhat_13_1/Assign-training/Adam/vhat_13_1/Read/ReadVariableOp:0(2training/Adam/vhat_13:08
�
training/Adam/vhat_14_1:0training/Adam/vhat_14_1/Assign-training/Adam/vhat_14_1/Read/ReadVariableOp:0(2training/Adam/vhat_14:08
�
training/Adam/vhat_15_1:0training/Adam/vhat_15_1/Assign-training/Adam/vhat_15_1/Read/ReadVariableOp:0(2training/Adam/vhat_15:08
�
training/Adam/vhat_16_1:0training/Adam/vhat_16_1/Assign-training/Adam/vhat_16_1/Read/ReadVariableOp:0(2training/Adam/vhat_16:08
�
training/Adam/vhat_17_1:0training/Adam/vhat_17_1/Assign-training/Adam/vhat_17_1/Read/ReadVariableOp:0(2training/Adam/vhat_17:08��Hu       ���	y戁_��A*

val_loss�(�?�qM        )��P	�爁_��A*

val_accuracy}��>^�N       �K"	�舁_��A*

loss[*@���)       ���	O鈁_��A*

accuracy��>�"�       ��2	�;0�_��A*

val_lossI��?�d�"       x=�	=0�_��A*

val_accuracyU�?��L�       ��-	�=0�_��A*

losskĬ?�_�|       ��2	}>0�_��A*

accuracyHm?���       ��2	C���_��A*

val_loss��S?�v�"       x=�	����_��A*

val_accuracy�l8?���       ��-	h���_��A*

loss�zn?���       ��2	���_��A*

accuracy��6?52N       ��2	�M�_��A*

val_loss��D?/>""       x=�	/O�_��A*

val_accuracy��F?�X�       ��-	P�_��A*

lossd�@?R]�       ��2	�P�_��A*

accuracy��D?D�4*       ��2	�J֌_��A*

val_loss��V?lB�"       x=�	jL֌_��A*

val_accuracy�B@?���       ��-	@M֌_��A*

loss�%?2��       ��2	�M֌_��A*

accuracy�wN?ԋ��       ��2	࿏_��A*

val_loss�� ?�j�"       x=�	�῏_��A*

val_accuracy$�O?��b       ��-	�⿏_��A*

lossS�?��y�       ��2	\㿏_��A*

accuracy~kT?�v�)       ��2	V���_��A*

val_loss�'?��@"       x=�	����_��A*

val_accuracy�Y?Y�d       ��-	����_��A*

lossC?���b       ��2	<���_��A*

accuracy�~W?	$��       ��2	����_��A*

val_loss�X�?_#��"       x=�	����_��A*

val_accuracyOd ?�ZG       ��-	Ù��_��A*

loss��>k4()       ��2	s���_��A*

accuracymyZ?w��H       ��2	v2��_��A*

val_loss�v?8��"       x=�	4��_��A*

val_accuracyi�V?�Ig       ��-	�4��_��A*

loss�l�>�WO9       ��2	�5��_��A*

accuracy}N]? �z�       ��2	+���_��A	*

val_loss��??'���"       x=�	����_��A	*

val_accuracy K?!�7�       ��-	����_��A	*

loss�S�>�-��       ��2	;���_��A	*

accuracy��_?��J       ��2	^��_��A
*

val_loss�?V�ǭ"       x=�	�_��_��A
*

val_accuracy�IT?���y       ��-	o`��_��A
*

lossC��>���~       ��2	a��_��A
*

accuracy��_?�|$       ��2	?7ȡ_��A*

val_lossz
�>�ק�"       x=�	�8ȡ_��A*

val_accuracy��_?`��+       ��-	�9ȡ_��A*

loss���>7�ɛ       ��2	`:ȡ_��A*

accuracy�a?�*%q       ��2	��Ǥ_��A*

val_loss�N?���"       x=�	�Ǥ_��A*

val_accuracy��H?A`��       ��-	�Ǥ_��A*

loss���>G˚       ��2	��Ǥ_��A*

accuracy&d?���       ��2	�!��_��A*

val_loss�&N?����"       x=�	#��_��A*

val_accuracy(�M?���~       ��-	�#��_��A*

loss_�>�Ap�       ��2	x$��_��A*

accuracyѶc?	�       ��2	�4��_��A*

val_loss�?��xO"       x=�	@6��_��A*

val_accuracy2dU? ���       ��-	7��_��A*

lossj��>�uJ       ��2	�7��_��A*

accuracy�<f?Hi:�       ��2	珜�_��A*

val_loss�?����"       x=�	����_��A*

val_accuracy/{W?= �       ��-	u���_��A*

loss�>��e�       ��2	-���_��A*

accuracy��g?nݑ�       ��2	����_��A*

val_loss?*���"       x=�	8���_��A*

val_accuracy5�W?��`$       ��-	���_��A*

loss�*�>1       ��2	����_��A*

accuracy��g?��;       ��2	G�_��A*

val_lossM?��D�"       x=�	��_��A*

val_accuracy��Z?X�~�       ��-	��_��A*

losslp�>�)�       ��2	W�_��A*

accuracy�Bi?�M��       ��2	��_��A*

val_loss���>��r�"       x=�	p��_��A*

val_accuracy}[\?�U�       ��-	F��_��A*

loss�Ô>P��       ��2	���_��A*

accuracy�qi?�Pa       ��2	�ƹ_��A*

val_lossU�>w���"       x=�	��ƹ_��A*

val_accuracy�c^?k��       ��-	r�ƹ_��A*

loss/��><��       ��2	�ƹ_��A*

accuracy2�h?g8n       ��2	����_��A*

val_lossT$?���"       x=�	h���_��A*

val_accuracy͎Z?E��A       ��-	)���_��A*

lossY��>��1�       ��2	͔��_��A*

accuracyRk?T�       ��2	��c�_��A*

val_loss+:?Qc�@"       x=�	R�c�_��A*

val_accuracy86Q?�U6       ��-	�c�_��A*

loss�]�>Z2N�       ��2	��c�_��A*

accuracyj?=��       ��2	��/�_��A*

val_loss�>u�"       x=�	)�/�_��A*

val_accuracy9Y?hY��       ��-	��/�_��A*

loss'�>#��        ��2	��/�_��A*

accuracy1�j?���       ��2	1���_��A*

val_loss$�>:�t�"       x=�	����_��A*

val_accuracyY�e?�XO       ��-	����_��A*

loss��>b��       ��2	9���_��A*

accuracy@�k? �%�       ��2	P76�_��A*

val_loss�X?��{"       x=�	96�_��A*

val_accuracy��X?��N       ��-	�96�_��A*

loss�/}>����       ��2	�:6�_��A*

accuracy��l?�
��       ��2	�+��_��A*

val_loss�U
?�9'"       x=�	�-��_��A*

val_accuracy�wX?��Yj       ��-	�.��_��A*

loss�zz>Y�:�       ��2	f/��_��A*

accuracy'm?a_       ��2	��_��A*

val_loss��>�N86"       x=�	���_��A*

val_accuracyyr^?v���       ��-	s��_��A*

loss�&}>���       ��2	B��_��A*

accuracy�m?���_       ��2	0,'�_��A*

val_loss��?lS��"       x=�	�-'�_��A*

val_accuracyv]W?���J       ��-	.'�_��A*

loss�o>cq�       ��2	4/'�_��A*

accuracy��m?E���       ��2	p��_��A*

val_lossk��>�<X>"       x=�	���_��A*

val_accuracy�*f?[s       ��-	���_��A*

lossf�c>���       ��2	J��_��A*

accuracy\�n?�&�       ��2	�i��_��A*

val_lossҚ�>�铙"       x=�	Ek��_��A*

val_accuracy�a?��W       ��-	l��_��A*

loss��p>�mD�       ��2	�l��_��A*

accuracyHn?+)e       ��2	��_��A*

val_loss��>�"       x=�	���_��A*

val_accuracyf?�Ѿ       ��-	���_��A*

lossb>���'       ��2	[��_��A*

accuracy��n?���/       ��2	����_��A*

val_lossN�?<�q"       x=�	r���_��A*

val_accuracy��^?��I       ��-	z���_��A*

loss��Y>���2       ��2	\���_��A*

accuracyD�o?��r�       ��2	� ��_��A *

val_loss1�?з[�"       x=�	l"��_��A *

val_accuracy�Z?K m       ��-	9#��_��A *

loss#�U>]��.       ��2	�#��_��A *

accuracy�o?���       ��2	���_��A!*

val_lossh�?�T�"       x=�	a��_��A!*

val_accuracyE2[?@8�       ��-	%��_��A!*

loss�	_>��"       ��2	���_��A!*

accuracy�n?���       ��2	���_��A"*

val_loss��>�zw�"       x=�	����_��A"*

val_accuracy��e?��}�       ��-	����_��A"*

lossQXP>4�^       ��2	U���_��A"*

accuracy�p?�y:       ��2	</I�_��A#*

val_loss�??I�ۊ"       x=�	1I�_��A#*

val_accuracy�O[?���z       ��-	2I�_��A#*

loss0U>���       ��2	�2I�_��A#*

accuracy��o?S	�       ��2	s���_��A$*

val_loss9��>�k��"       x=�	3���_��A$*

val_accuracyF^d?�P��       ��-	3���_��A$*

loss��F>�X��       ��2	���_��A$*

accuracyC�p?)�d�       ��2	�@7�_��A%*

val_loss��>���"       x=�	tB7�_��A%*

val_accuracy/�`?�|��       ��-	|C7�_��A%*

loss��G>նxe       ��2	^D7�_��A%*

accuracy�p?�K�T       ��2	����_��A&*

val_loss�h�>4#*"       x=�	L���_��A&*

val_accuracy��d?���m       ��-	���_��A&*

lossZ�E>�Ԧ       ��2	����_��A&*

accuracy-�p?K	       ��2	��&�_��A'*

val_loss�S�>i���"       x=�	&�&�_��A'*

val_accuracyzc?x�0       ��-	��&�_��A'*

lossomA>��]       ��2	��&�_��A'*

accuracy��q?���E       ��2	�j"�_��A(*

val_lossV�>L�"       x=�	Yl"�_��A(*

val_accuracy�Hf?G|�u       ��-	'm"�_��A(*

lossѵ:>E2/�       ��2	�m"�_��A(*

accuracyZ�q?�XB       ��2	z�<�_��A)*

val_lossF	?X}*"       x=�	�<�_��A)*

val_accuracy_�\?��P       ��-	ٯ<�_��A)*

lossd�4>��       ��2	��<�_��A)*

accuracywKr?r,9A       ��2	H�<`��A**

val_loss�9�>���"       x=�	��<`��A**

val_accuracy�d?V��i       ��-	��<`��A**

loss�a=>[Cmo       ��2	C�<`��A**

accuracy��q?���v       ��2	T�r`��A+*

val_loss`A?#V�"       x=�	��r`��A+*

val_accuracy�f]?����       ��-	��r`��A+*

loss̀4>�^�       ��2	u�r`��A+*

accuracy��q?�9�       ��2	_'�`��A,*

val_loss���>����"       x=�	)�`��A,*

val_accuracy��e?�E       ��-	
*�`��A,*

lossmb1>9�Uf       ��2	�*�`��A,*

accuracy��r?�tK�       ��2	T�J`��A-*

val_loss�?�W��"       x=�	�J`��A-*

val_accuracyS�`?�7J       ��-	ՒJ`��A-*

loss,�*>u���       ��2	��J`��A-*

accuracy��r?����       ��2	�F�`��A.*

val_lossBR ?��ob"       x=�	�I�`��A.*

val_accuracy�a?����       ��-	�J�`��A.*

lossJ�5>���#       ��2	�K�`��A.*

accuracy�'r?�%�       ��2	e��`��A/*

val_loss|��>h��"       x=�	���`��A/*

val_accuracy��b?�       ��-	���`��A/*

loss��*>�j�       ��2	}��`��A/*

accuracyUs?P�F9       ��2	��`��A0*

val_lossF��>w@�"       x=�	���`��A0*

val_accuracy�a? ?�R       ��-	]��`��A0*

lossS�)>B9S       ��2		��`��A0*

accuracym�r?	�"       ��2	���`��A1*

val_loss��?
<,j"       x=�	:�`��A1*

val_accuracy"�_?{-�T       ��-	�`��A1*

loss��)>�U��       ��2	��`��A1*

accuracy�s?�&��       ��2	2)`��A2*

val_loss�%�>Kc_"       x=�	�	)`��A2*

val_accuracyLe?_       ��-	�
)`��A2*

loss�">�?t�       ��2	�)`��A2*

accuracy�xs?x9a�       ��2	�v�`��A3*

val_loss<;?י��"       x=�	�x�`��A3*

val_accuracyv�[?�a(1       ��-	yy�`��A3*

loss%>`��:       ��2	Jz�`��A3*

accuracy�:s?AF�       ��2	q^#`��A4*

val_loss�g�>W=�"       x=�	~s^#`��A4*

val_accuracys�b?t�+_       ��-	pt^#`��A4*

loss��>yGl       ��2	?u^#`��A4*

accuracy�s?<���       ��2	�	�&`��A5*

val_loss%�>����"       x=�	4�&`��A5*

val_accuracyf?�&��       ��-	�&`��A5*

loss�>V��       ��2	��&`��A5*

accuracy3�s?Bҭ       ��2	P]*`��A6*

val_loss?�&��"       x=�	�]*`��A6*

val_accuracyN`?_7��       ��-	�]*`��A6*

lossGo>#dg       ��2	O]*`��A6*

accuracy��s?Z�=       ��2	���-`��A7*

val_loss�G?���r"       x=�		��-`��A7*

val_accuracyN`?:"�       ��-	Ŋ�-`��A7*

loss֞ > "3�       ��2	j��-`��A7*

accuracy��s?�腏       ��2	�z0`��A8*

val_lossHF�>0˙�"       x=�	mz0`��A8*

val_accuracyf?�Y       ��-	Pz0`��A8*

loss@F>�5�       ��2	z0`��A8*

accuracyUt?��k       ��2		S�3`��A9*

val_lossx�?�DH�"       x=�	�T�3`��A9*

val_accuracy2&c?���	       ��-	~U�3`��A9*

lossC_>N��       ��2	DV�3`��A9*

accuracy�>u?�u�       ��2	��6`��A:*

val_loss�?H��g"       x=�	4�6`��A:*

val_accuracy6a?Sd�       ��-	�6`��A:*

loss��>b�:       ��2	��6`��A:*

accuracy��s?�_/�       ��2	]�9`��A;*

val_loss���>2t�Z"       x=�	�^�9`��A;*

val_accuracy�ac?[�       ��-	p_�9`��A;*

lossk�>n�P�       ��2	`�9`��A;*

accuracyz�t?�^�O       ��2	R(�<`��A<*

val_loss<^?R=&�"       x=�	�)�<`��A<*

val_accuracyeb?�^��       ��-	E*�<`��A<*

loss�>���       ��2	�*�<`��A<*

accuracy�~t?<��z       ��2	x��?`��A=*

val_loss���>s��"       x=�	��?`��A=*

val_accuracy�b?RP"N       ��-	ܹ�?`��A=*

loss�#>"��       ��2	���?`��A=*

accuracy�t?�x�       ��2	!��B`��A>*

val_loss���>q�s"       x=�	۫�B`��A>*

val_accuracy9$h?@q�X       ��-	���B`��A>*

lossƏ>���       ��2	u��B`��A>*

accuracy��t?*��       ��2	D#F`��A?*

val_loss���>:9�u"       x=�	�E#F`��A?*

val_accuracy�*f?�`       ��-	�F#F`��A?*

loss�>�n�3       ��2	�G#F`��A?*

accuracy� u?�U�'       ��2	\��I`��A@*

val_loss;8�>�W��"       x=�	G��I`��A@*

val_accuracyeb?g͍p       ��-	W��I`��A@*

loss��>��|�       ��2	:��I`��A@*

accuracy݉t?˓�       ��2	���L`��AA*

val_lossr��>�1��"       x=�	I��L`��AA*

val_accuracy,�b?�hQ�       ��-	'��L`��AA*

loss��>=Op       ��2	���L`��AA*

accuracy�>u?�ځ       ��2	���O`��AB*

val_loss��>�êH"       x=�	-��O`��AB*

val_accuracy�*f?��,�       ��-	��O`��AB*

loss��>��       ��2	���O`��AB*

accuracy�u?�Y�       ��2	j�R`��AC*

val_loss��?�|��"       x=�	�R`��AC*

val_accuracy7^?��e       ��-	��R`��AC*

lossOc>3��{       ��2	��R`��AC*

accuracy7u?�Xa�       ��2	Ͻ�U`��AD*

val_loss�� ?�+<�"       x=�	@��U`��AD*

val_accuracyۊd?X���       ��-	��U`��AD*

loss'6	>LO       ��2	���U`��AD*

accuracyg�u?�݄�       ��2	�ȔX`��AE*

val_loss���>`��"       x=�	uʔX`��AE*

val_accuracy2&c?O/�[       ��-	B˔X`��AE*

loss�	�=k�S�       ��2	�˔X`��AE*

accuracy�Kv?	���       ��2	�+l[`��AF*

val_loss��?u�{�"       x=�	^-l[`��AF*

val_accuracy?`_?� �F       ��-	(.l[`��AF*

loss�>��B�       ��2	�.l[`��AF*

accuracys�u?ɟ�W       ��2	�^`��AG*

val_loss�~.?�
�"       x=�	[�^`��AG*

val_accuracy͎Z?�u��       ��-	(�^`��AG*

loss�
�=�b��       ��2	Й^`��AG*

accuracy�bv?�M        ��2	�(�``��AH*

val_loss��>;Yh"       x=�	4*�``��AH*

val_accuracy�c?��/       ��-	+�``��AH*

loss'>��       ��2	�+�``��AH*

accuracydu?{1��       ��2	$B�c`��AI*

val_loss��?��V�"       x=�	�C�c`��AI*

val_accuracy2&c?���@       ��-	hD�c`��AI*

lossy��=IU%�       ��2	E�c`��AI*

accuracy��u?�ꩇ       ��2	�;�f`��AJ*

val_loss��?]:ܳ"       x=�	.=�f`��AJ*

val_accuracy5c?خ��       ��-	>�f`��AJ*

loss&R>���/       ��2	�>�f`��AJ*

accuracyV�u?�j��       ��2	��{i`��AK*

val_lossL�>,ȯs"       x=�	��{i`��AK*

val_accuracy�Ze?���       ��-	P�{i`��AK*

loss���=Hn�       ��2	�{i`��AK*

accuracywv?���       ��2	:�)l`��AL*

val_loss�?, �"       x=�	��)l`��AL*

val_accuracyѣa?rK�       ��-	F�)l`��AL*

lossO��=���`       ��2	��)l`��AL*

accuracy�@v?����       ��2	'l�n`��AM*

val_loss�A�>u��"       x=�	�m�n`��AM*

val_accuracyL�d?E��       ��-	Xn�n`��AM*

lossI7>(��{       ��2	o�n`��AM*

accuracyYv?B�P�       ��2	�\�q`��AN*

val_lossg{�>����"       x=�	F^�q`��AN*

val_accuracyd?�,��       ��-	6_�q`��AN*

loss��=���Q       ��2	�_�q`��AN*

accuracy��v?փM|       ��2	rm�t`��AO*

val_loss�'?�KD]"       x=�	�n�t`��AO*

val_accuracy	Z?�� ~       ��-	�o�t`��AO*

loss�>��t�       ��2	�p�t`��AO*

accuracy�u?+>�       ��2	�Ϊw`��AP*

val_loss��>7<��"       x=�	tЪw`��AP*

val_accuracy�e?���'       ��-	AѪw`��AP*

loss��=X� H       ��2	�Ѫw`��AP*

accuracy�v?w��i       ��2	�+rz`��AQ*

val_loss��?1��Z"       x=�	
-rz`��AQ*

val_accuracys�b?UB�v       ��-	�-rz`��AQ*

loss@"�=J'�Y       ��2	f.rz`��AQ*

accuracy:�v?���       ��2	��4}`��AR*

val_loss�N?02�	"       x=�	
�4}`��AR*

val_accuracy��d?��$       ��-	��4}`��AR*

lossj��=�_��       ��2	|�4}`��AR*

accuracy�v?�W�       ��2	��
�`��AS*

val_loss��>�4��"       x=�	c�
�`��AS*

val_accuracyL�d?�k�       ��-	 �
�`��AS*

lossF��=��       ��2	ĵ
�`��AS*

accuracy[v?�'r�       ��2	b���`��AT*

val_loss�"*?�Q"       x=�	���`��AT*

val_accuracy�L\?�;Fh       ��-	����`��AT*

loss�]�=��'O       ��2	a���`��AT*

accuracy!�v?����       ��2	N)ԅ`��AU*

val_loss���>�.�v"       x=�	�*ԅ`��AU*

val_accuracy��f?CȠ       ��-	�+ԅ`��AU*

loss��=�J*�       ��2	=,ԅ`��AU*

accuracyTfv?�ȍ       ��2	@ӈ`��AV*

val_loss9�?s�f�"       x=�	�ӈ`��AV*

val_accuracy,�b?�,�       ��-	�ӈ`��AV*

loss���=�䈰       ��2	nӈ`��AV*

accuracy��v?��P       ��2	O�Ջ`��AW*

val_loss���>�C�"       x=�	��Ջ`��AW*

val_accuracy]h?�PG       ��-	��Ջ`��AW*

loss�a�=�-�       ��2	:�Ջ`��AW*

accuracy�v?d���       ��2	��Ў`��AX*

val_loss�?����"       x=�	^�Ў`��AX*

val_accuracy�pc?Y�ؾ       ��-	A�Ў`��AX*

loss���=���       ��2	�Ў`��AX*

accuracy:�v?_Nq�       ��2	]��`��AY*

val_loss�-?6�� "       x=�		��`��AY*

val_accuracy��X?�X�       ��-	���`��AY*

loss�5�=H��       ��2	֩�`��AY*

accuracy��v?Ld[d       ��2	!P�`��AZ*

val_loss߯�>��ػ"       x=�	�P�`��AZ*

val_accuracy}�e?��\�       ��-	�P�`��AZ*

loss�}�=�@j       ��2	lP�`��AZ*

accuracy2�v?�X�       ��2	�(��`��A[*

val_loss���>T׳"       x=�	k*��`��A[*

val_accuracy��c?X���       ��-	V+��`��A[*

loss�U�=��G       ��2	',��`��A[*

accuracyɱw?��       ��2	�Xԛ`��A\*

val_loss8r?�)��"       x=�	SZԛ`��A\*

val_accuracy�;a?+�:�       ��-	>[ԛ`��A\*

loss���=ǻ�       ��2	\ԛ`��A\*

accuracy��v?[4�       ��2	R*��`��A]*

val_lossTH�>��h"       x=�	�+��`��A]*

val_accuracy��i?>;o       ��-	�,��`��A]*

lossM`�=�ʹ�       ��2	x-��`��A]*

accuracy�w?�R�a       ��2	����`��A^*

val_loss�>s�7"       x=�	T���`��A^*

val_accuracy�f?ť�'       ��-	&���`��A^*

loss���=���4       ��2	����`��A^*

accuracylw?��X�       ��2	�ߤ`��A_*

val_loss��?�bqd"       x=�	{�ߤ`��A_*

val_accuracy�b?-i��       ��-	U�ߤ`��A_*

loss���=��[p       ��2	�ߤ`��A_*

accuracy��w?����       ��2	8��`��A`*

val_loss�?� $�"       x=�	���`��A`*

val_accuracyS�`?��       ��-	k��`��A`*

loss��=3��%       ��2	��`��A`*

accuracyb�w?8Xۯ       ��2	��w�`��Aa*

val_lossKQ?�z�D"       x=�	*�w�`��Aa*

val_accuracyp!`? "hV       ��-	��w�`��Aa*

loss$-�=i$]n       ��2	��w�`��Aa*

accuracyx�w?���       ��2	��D�`��Ab*

val_loss���>Gv��"       x=�	M�D�`��Ab*

val_accuracy3�g?O�.       ��-	�D�`��Ab*

loss�$�=�F��       ��2	��D�`��Ab*

accuracy6�v?��{�       ��2	��`��Ac*

val_loss� ?8�~#"       x=�	�`��Ac*

val_accuracyLe?g�Vb       ��-	��`��Ac*

losst��=2�       ��2	��`��Ac*

accuracys�w?.[g�