       �K"	  @٠��Abrain.Event:2a8X��     �#�	I]٠��A"��
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
$dense_1/random_uniform/RandomUniformRandomUniformdense_1/random_uniform/shape*
seed���)*
T0*
dtype0* 
_output_shapes
:
��*
seed2���
z
dense_1/random_uniform/subSubdense_1/random_uniform/maxdense_1/random_uniform/min*
T0*
_output_shapes
: 
�
dense_1/random_uniform/mulMul$dense_1/random_uniform/RandomUniformdense_1/random_uniform/sub* 
_output_shapes
:
��*
T0
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
dense_1/ReluReludense_1/BiasAdd*(
_output_shapes
:����������*
T0
j
batch_normalization_1/ConstConst*
dtype0*
_output_shapes	
:�*
valueB�*  �?
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
batch_normalization_1/Const_1Const*
dtype0*
_output_shapes	
:�*
valueB�*    
�
batch_normalization_1/betaVarHandleOp*+
shared_namebatch_normalization_1/beta*-
_class#
!loc:@batch_normalization_1/beta*
	container *
shape:�*
dtype0*
_output_shapes
: 
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
!batch_normalization_1/moving_meanVarHandleOp*
	container *
shape:�*
dtype0*
_output_shapes
: *2
shared_name#!batch_normalization_1/moving_mean*4
_class*
(&loc:@batch_normalization_1/moving_mean
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
_output_shapes
:	�*
	keep_dims(*

Tidx0*
T0
�
*batch_normalization_1/moments/StopGradientStopGradient"batch_normalization_1/moments/mean*
_output_shapes
:	�*
T0
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
%batch_normalization_1/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
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
#batch_normalization_1/strided_sliceStridedSlicebatch_normalization_1/Shape)batch_normalization_1/strided_slice/stack+batch_normalization_1/strided_slice/stack_1+batch_normalization_1/strided_slice/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0*
shrink_axis_mask
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
!batch_normalization_1/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
batch_normalization_1/rangeRange!batch_normalization_1/range/startbatch_normalization_1/Rank!batch_normalization_1/range/delta*

Tidx0*
_output_shapes
:
�
 batch_normalization_1/Prod/inputPack#batch_normalization_1/strided_slice*
T0*

axis *
N*
_output_shapes
:
�
batch_normalization_1/ProdProd batch_normalization_1/Prod/inputbatch_normalization_1/range*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
~
batch_normalization_1/CastCastbatch_normalization_1/Prod*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
`
batch_normalization_1/sub/yConst*
valueB
 *� �?*
dtype0*
_output_shapes
: 
z
batch_normalization_1/subSubbatch_normalization_1/Castbatch_normalization_1/sub/y*
_output_shapes
: *
T0
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
batch_normalization_1/Const_4Const*
dtype0*
_output_shapes
: *
valueB
 *
�#<*4
_class*
(&loc:@batch_normalization_1/moving_mean
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
batch_normalization_1/mul_1Mulbatch_normalization_1/sub_1batch_normalization_1/Const_4*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes	
:�
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
batch_normalization_1/Const_5Const*
valueB
 *
�#<*8
_class.
,*loc:@batch_normalization_1/moving_variance*
dtype0*
_output_shapes
: 
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
"batch_normalization_1/cond/pred_idIdentitykeras_learning_phase*
T0
*
_output_shapes
: 
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
:batch_normalization_1/cond/batchnorm/ReadVariableOp/SwitchSwitch%batch_normalization_1/moving_variance"batch_normalization_1/cond/pred_id*
_output_shapes
: : *
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance
�
*batch_normalization_1/cond/batchnorm/add/yConst$^batch_normalization_1/cond/switch_f*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
(batch_normalization_1/cond/batchnorm/addAddV23batch_normalization_1/cond/batchnorm/ReadVariableOp*batch_normalization_1/cond/batchnorm/add/y*
_output_shapes	
:�*
T0
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
(batch_normalization_1/cond/batchnorm/mulMul*batch_normalization_1/cond/batchnorm/Rsqrt7batch_normalization_1/cond/batchnorm/mul/ReadVariableOp*
_output_shapes	
:�*
T0
�
*batch_normalization_1/cond/batchnorm/mul_1Mul1batch_normalization_1/cond/batchnorm/mul_1/Switch(batch_normalization_1/cond/batchnorm/mul*
T0*(
_output_shapes
:����������
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
<batch_normalization_1/cond/batchnorm/ReadVariableOp_1/SwitchSwitch!batch_normalization_1/moving_mean"batch_normalization_1/cond/pred_id*
_output_shapes
: : *
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean
�
*batch_normalization_1/cond/batchnorm/mul_2Mul5batch_normalization_1/cond/batchnorm/ReadVariableOp_1(batch_normalization_1/cond/batchnorm/mul*
T0*
_output_shapes	
:�
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
N**
_output_shapes
:����������: *
T0
m
dense_2/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"�   �   
_
dense_2/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *q��
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
seed2���*
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
dtype0*
_output_shapes
: *
shared_namedense_2/bias*
_class
loc:@dense_2/bias*
	container *
shape:�
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
batch_normalization_2/betaVarHandleOp*
dtype0*
_output_shapes
: *+
shared_namebatch_normalization_2/beta*-
_class#
!loc:@batch_normalization_2/beta*
	container *
shape:�
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
&batch_normalization_2/moments/varianceMean/batch_normalization_2/moments/SquaredDifference8batch_normalization_2/moments/variance/reduction_indices*
T0*
_output_shapes
:	�*
	keep_dims(*

Tidx0
�
%batch_normalization_2/moments/SqueezeSqueeze"batch_normalization_2/moments/mean*
_output_shapes	
:�*
squeeze_dims
 *
T0
�
'batch_normalization_2/moments/Squeeze_1Squeeze&batch_normalization_2/moments/variance*
T0*
_output_shapes	
:�*
squeeze_dims
 
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
%batch_normalization_2/batchnorm/RsqrtRsqrt#batch_normalization_2/batchnorm/add*
_output_shapes	
:�*
T0
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
%batch_normalization_2/batchnorm/mul_2Mul%batch_normalization_2/moments/Squeeze#batch_normalization_2/batchnorm/mul*
_output_shapes	
:�*
T0
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
+batch_normalization_2/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
u
+batch_normalization_2/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
#batch_normalization_2/strided_sliceStridedSlicebatch_normalization_2/Shape)batch_normalization_2/strided_slice/stack+batch_normalization_2/strided_slice/stack_1+batch_normalization_2/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
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
batch_normalization_2/rangeRange!batch_normalization_2/range/startbatch_normalization_2/Rank!batch_normalization_2/range/delta*
_output_shapes
:*

Tidx0
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
batch_normalization_2/Const_4Const*
valueB
 *
�#<*4
_class*
(&loc:@batch_normalization_2/moving_mean*
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
)batch_normalization_2/AssignSubVariableOpAssignSubVariableOp!batch_normalization_2/moving_meanbatch_normalization_2/mul_1*4
_class*
(&loc:@batch_normalization_2/moving_mean*
dtype0
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
batch_normalization_2/mul_2Mulbatch_normalization_2/sub_2batch_normalization_2/Const_5*
_output_shapes	
:�*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance
�
+batch_normalization_2/AssignSubVariableOp_1AssignSubVariableOp%batch_normalization_2/moving_variancebatch_normalization_2/mul_2*8
_class.
,*loc:@batch_normalization_2/moving_variance*
dtype0
�
&batch_normalization_2/ReadVariableOp_3ReadVariableOp%batch_normalization_2/moving_variance,^batch_normalization_2/AssignSubVariableOp_1*8
_class.
,*loc:@batch_normalization_2/moving_variance*
dtype0*
_output_shapes	
:�
z
!batch_normalization_2/cond/SwitchSwitchkeras_learning_phasekeras_learning_phase*
T0
*
_output_shapes
: : 
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
1batch_normalization_2/cond/batchnorm/mul_1/SwitchSwitchdense_2/Relu"batch_normalization_2/cond/pred_id*<
_output_shapes*
(:����������:����������*
T0*
_class
loc:@dense_2/Relu
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
dropout_1/cond/dropout/ShapeShape%dropout_1/cond/dropout/Shape/Switch:1*
T0*
out_type0*
_output_shapes
:
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
dtype0*(
_output_shapes
:����������*
seed2���*
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
#dropout_1/cond/dropout/GreaterEqualGreaterEqual%dropout_1/cond/dropout/random_uniformdropout_1/cond/dropout/rate*(
_output_shapes
:����������*
T0
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
dense_3/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *q��
_
dense_3/random_uniform/maxConst*
valueB
 *q�>*
dtype0*
_output_shapes
: 
�
$dense_3/random_uniform/RandomUniformRandomUniformdense_3/random_uniform/shape*
dtype0* 
_output_shapes
:
��*
seed2���*
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
dense_3/biasVarHandleOp*
shape:�*
dtype0*
_output_shapes
: *
shared_namedense_3/bias*
_class
loc:@dense_3/bias*
	container 
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
T0*(
_output_shapes
:����������*
transpose_a( 
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
batch_normalization_3/gammaVarHandleOp*
shape:�*
dtype0*
_output_shapes
: *,
shared_namebatch_normalization_3/gamma*.
_class$
" loc:@batch_normalization_3/gamma*
	container 
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
batch_normalization_3/Const_1Const*
dtype0*
_output_shapes	
:�*
valueB�*    
�
batch_normalization_3/betaVarHandleOp*
	container *
shape:�*
dtype0*
_output_shapes
: *+
shared_namebatch_normalization_3/beta*-
_class#
!loc:@batch_normalization_3/beta
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
!batch_normalization_3/moving_meanVarHandleOp*
shape:�*
dtype0*
_output_shapes
: *2
shared_name#!batch_normalization_3/moving_mean*4
_class*
(&loc:@batch_normalization_3/moving_mean*
	container 
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
&batch_normalization_3/moments/varianceMean/batch_normalization_3/moments/SquaredDifference8batch_normalization_3/moments/variance/reduction_indices*
	keep_dims(*

Tidx0*
T0*
_output_shapes
:	�
�
%batch_normalization_3/moments/SqueezeSqueeze"batch_normalization_3/moments/mean*
_output_shapes	
:�*
squeeze_dims
 *
T0
�
'batch_normalization_3/moments/Squeeze_1Squeeze&batch_normalization_3/moments/variance*
_output_shapes	
:�*
squeeze_dims
 *
T0
j
%batch_normalization_3/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *o�:
�
#batch_normalization_3/batchnorm/addAddV2'batch_normalization_3/moments/Squeeze_1%batch_normalization_3/batchnorm/add/y*
_output_shapes	
:�*
T0
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
#batch_normalization_3/batchnorm/mulMul%batch_normalization_3/batchnorm/Rsqrt2batch_normalization_3/batchnorm/mul/ReadVariableOp*
_output_shapes	
:�*
T0
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
batch_normalization_3/ShapeShapedense_3/Relu*
_output_shapes
:*
T0*
out_type0
s
)batch_normalization_3/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
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
N*
_output_shapes
:*
T0*

axis 
\
batch_normalization_3/RankConst*
value	B :*
dtype0*
_output_shapes
: 
c
!batch_normalization_3/range/startConst*
dtype0*
_output_shapes
: *
value	B : 
c
!batch_normalization_3/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
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
batch_normalization_3/ProdProd batch_normalization_3/Prod/inputbatch_normalization_3/range*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
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
batch_normalization_3/truedivRealDivbatch_normalization_3/Castbatch_normalization_3/sub*
_output_shapes
: *
T0
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
&batch_normalization_3/ReadVariableOp_1ReadVariableOp!batch_normalization_3/moving_mean*^batch_normalization_3/AssignSubVariableOp*4
_class*
(&loc:@batch_normalization_3/moving_mean*
dtype0*
_output_shapes	
:�
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
&batch_normalization_3/ReadVariableOp_3ReadVariableOp%batch_normalization_3/moving_variance,^batch_normalization_3/AssignSubVariableOp_1*8
_class.
,*loc:@batch_normalization_3/moving_variance*
dtype0*
_output_shapes	
:�
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
:batch_normalization_3/cond/batchnorm/ReadVariableOp/SwitchSwitch%batch_normalization_3/moving_variance"batch_normalization_3/cond/pred_id*
_output_shapes
: : *
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance
�
*batch_normalization_3/cond/batchnorm/add/yConst$^batch_normalization_3/cond/switch_f*
valueB
 *o�:*
dtype0*
_output_shapes
: 
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
dropout_2/cond/dropout/ShapeShape%dropout_2/cond/dropout/Shape/Switch:1*
_output_shapes
:*
T0*
out_type0
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
3dropout_2/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_2/cond/dropout/Shape*
T0*
dtype0*(
_output_shapes
:����������*
seed2���*
seed���)
�
)dropout_2/cond/dropout/random_uniform/subSub)dropout_2/cond/dropout/random_uniform/max)dropout_2/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
�
)dropout_2/cond/dropout/random_uniform/mulMul3dropout_2/cond/dropout/random_uniform/RandomUniform)dropout_2/cond/dropout/random_uniform/sub*(
_output_shapes
:����������*
T0
�
%dropout_2/cond/dropout/random_uniformAdd)dropout_2/cond/dropout/random_uniform/mul)dropout_2/cond/dropout/random_uniform/min*
T0*(
_output_shapes
:����������
{
dropout_2/cond/dropout/sub/xConst^dropout_2/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  �?
}
dropout_2/cond/dropout/subSubdropout_2/cond/dropout/sub/xdropout_2/cond/dropout/rate*
T0*
_output_shapes
: 

 dropout_2/cond/dropout/truediv/xConst^dropout_2/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  �?
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
dense_4/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *q��
_
dense_4/random_uniform/maxConst*
valueB
 *q�>*
dtype0*
_output_shapes
: 
�
$dense_4/random_uniform/RandomUniformRandomUniformdense_4/random_uniform/shape*
T0*
dtype0* 
_output_shapes
:
��*
seed2��h*
seed���)
z
dense_4/random_uniform/subSubdense_4/random_uniform/maxdense_4/random_uniform/min*
_output_shapes
: *
T0
�
dense_4/random_uniform/mulMul$dense_4/random_uniform/RandomUniformdense_4/random_uniform/sub* 
_output_shapes
:
��*
T0
�
dense_4/random_uniformAdddense_4/random_uniform/muldense_4/random_uniform/min*
T0* 
_output_shapes
:
��
�
dense_4/kernelVarHandleOp*
shared_namedense_4/kernel*!
_class
loc:@dense_4/kernel*
	container *
shape:
��*
dtype0*
_output_shapes
: 
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
batch_normalization_4/ConstConst*
dtype0*
_output_shapes	
:�*
valueB�*  �?
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
batch_normalization_4/Const_1Const*
valueB�*    *
dtype0*
_output_shapes	
:�
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
batch_normalization_4/Const_2Const*
valueB�*    *
dtype0*
_output_shapes	
:�
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
"batch_normalization_4/moments/meanMeandense_4/Relu4batch_normalization_4/moments/mean/reduction_indices*
_output_shapes
:	�*
	keep_dims(*

Tidx0*
T0
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
&batch_normalization_4/moments/varianceMean/batch_normalization_4/moments/SquaredDifference8batch_normalization_4/moments/variance/reduction_indices*
	keep_dims(*

Tidx0*
T0*
_output_shapes
:	�
�
%batch_normalization_4/moments/SqueezeSqueeze"batch_normalization_4/moments/mean*
_output_shapes	
:�*
squeeze_dims
 *
T0
�
'batch_normalization_4/moments/Squeeze_1Squeeze&batch_normalization_4/moments/variance*
squeeze_dims
 *
T0*
_output_shapes	
:�
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
N*
_output_shapes
:*
T0*

axis 
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
batch_normalization_4/Const_4Const*
valueB
 *
�#<*4
_class*
(&loc:@batch_normalization_4/moving_mean*
dtype0*
_output_shapes
: 
�
$batch_normalization_4/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
dtype0*
_output_shapes	
:�
�
batch_normalization_4/sub_1Sub$batch_normalization_4/ReadVariableOp%batch_normalization_4/moments/Squeeze*
T0*4
_class*
(&loc:@batch_normalization_4/moving_mean*
_output_shapes	
:�
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
batch_normalization_4/Const_5Const*
valueB
 *
�#<*8
_class.
,*loc:@batch_normalization_4/moving_variance*
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
!batch_normalization_4/cond/SwitchSwitchkeras_learning_phasekeras_learning_phase*
_output_shapes
: : *
T0

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
*batch_normalization_4/cond/batchnorm/RsqrtRsqrt(batch_normalization_4/cond/batchnorm/add*
_output_shapes	
:�*
T0
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
T0*
N**
_output_shapes
:����������: 
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
dense_5/random_uniform/maxConst*
valueB
 *�?>*
dtype0*
_output_shapes
: 
�
$dense_5/random_uniform/RandomUniformRandomUniformdense_5/random_uniform/shape*
T0*
dtype0*
_output_shapes
:	�+*
seed2���*
seed���)
z
dense_5/random_uniform/subSubdense_5/random_uniform/maxdense_5/random_uniform/min*
_output_shapes
: *
T0
�
dense_5/random_uniform/mulMul$dense_5/random_uniform/RandomUniformdense_5/random_uniform/sub*
T0*
_output_shapes
:	�+

dense_5/random_uniformAdddense_5/random_uniform/muldense_5/random_uniform/min*
T0*
_output_shapes
:	�+
�
dense_5/kernelVarHandleOp*
shape:	�+*
dtype0*
_output_shapes
: *
shared_namedense_5/kernel*!
_class
loc:@dense_5/kernel*
	container 
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
)Adam/iterations/Initializer/initial_valueConst*
value	B	 R *"
_class
loc:@Adam/iterations*
dtype0	*
_output_shapes
: 
�
Adam/iterationsVarHandleOp*
	container *
shape: *
dtype0	*
_output_shapes
: * 
shared_nameAdam/iterations*"
_class
loc:@Adam/iterations
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
shape: *
dtype0*
_output_shapes
: *
shared_nameAdam/beta_1*
_class
loc:@Adam/beta_1*
	container 
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
$Adam/decay/Initializer/initial_valueConst*
valueB
 *    *
_class
loc:@Adam/decay*
dtype0*
_output_shapes
: 
�

Adam/decayVarHandleOp*
shape: *
dtype0*
_output_shapes
: *
shared_name
Adam/decay*
_class
loc:@Adam/decay*
	container 
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
totalVarHandleOp*
shared_nametotal*
_class

loc:@total*
	container *
shape: *
dtype0*
_output_shapes
: 
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
#metrics/accuracy/ArgMax_1/dimensionConst*
dtype0*
_output_shapes
: *
valueB :
���������
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
metrics/accuracy/SumSummetrics/accuracy/Castmetrics/accuracy/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
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
metrics/accuracy/Cast_1Castmetrics/accuracy/Size*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0
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
metrics/accuracy/truedivRealDiv!metrics/accuracy/ReadVariableOp_2'metrics/accuracy/truediv/ReadVariableOp*
_output_shapes
: *
T0
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
N*
_output_shapes
:*

Tidx0*
T0
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
N*
_output_shapes
:*
T0*

axis 
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
Uloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/concat_1ConcatV2^loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/concat_1/values_0Tloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Slice_1Zloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/concat_1/axis*
N*
_output_shapes
:*

Tidx0*
T0
�
Vloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape_1Reshapedense_5_targetUloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/concat_1*
T0*
Tshape0*0
_output_shapes
:������������������
�
Lloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogitsTloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/ReshapeVloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape_1*?
_output_shapes-
+:���������:������������������*
T0
�
Tloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Sub_2/yConst*
dtype0*
_output_shapes
: *
value	B :
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
: *
	keep_dims( *

Tidx0
�
Jloss/dense_5_loss/categorical_crossentropy/weighted_loss/num_elements/SizeSize<loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul*
_output_shapes
: *
T0*
out_type0
�
Jloss/dense_5_loss/categorical_crossentropy/weighted_loss/num_elements/CastCastJloss/dense_5_loss/categorical_crossentropy/weighted_loss/num_elements/Size*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
�
@loss/dense_5_loss/categorical_crossentropy/weighted_loss/truedivRealDiv<loss/dense_5_loss/categorical_crossentropy/weighted_loss/SumJloss/dense_5_loss/categorical_crossentropy/weighted_loss/num_elements/Cast*
_output_shapes
: *
T0
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
training/Adam/gradients/ShapeConst*
valueB *
_class
	loc:@Mean*
dtype0*
_output_shapes
: 

!training/Adam/gradients/grad_ys_0Const*
dtype0*
_output_shapes
: *
valueB
 *  �?*
_class
	loc:@Mean
�
training/Adam/gradients/FillFilltraining/Adam/gradients/Shape!training/Adam/gradients/grad_ys_0*
T0*

index_type0*
_class
	loc:@Mean*
_output_shapes
: 
�
/training/Adam/gradients/Mean_grad/Reshape/shapeConst*
dtype0*
_output_shapes
: *
valueB *
_class
	loc:@Mean
�
)training/Adam/gradients/Mean_grad/ReshapeReshapetraining/Adam/gradients/Fill/training/Adam/gradients/Mean_grad/Reshape/shape*
T0*
Tshape0*
_class
	loc:@Mean*
_output_shapes
: 
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
)training/Adam/gradients/Mean_grad/Const_1Const*
valueB
 *  �?*
_class
	loc:@Mean*
dtype0*
_output_shapes
: 
�
)training/Adam/gradients/Mean_grad/truedivRealDiv&training/Adam/gradients/Mean_grad/Tile)training/Adam/gradients/Mean_grad/Const_1*
_output_shapes
: *
T0*
_class
	loc:@Mean
�
)training/Adam/gradients/loss/mul_grad/MulMul)training/Adam/gradients/Mean_grad/truediv@loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv*
_output_shapes
: *
T0*
_class
loc:@loss/mul
�
+training/Adam/gradients/loss/mul_grad/Mul_1Mul)training/Adam/gradients/Mean_grad/truediv
loss/mul/x*
T0*
_class
loc:@loss/mul*
_output_shapes
: 
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
straining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsctraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/Shapeetraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0*S
_classI
GEloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv
�
etraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/RealDivRealDiv+training/Adam/gradients/loss/mul_grad/Mul_1Jloss/dense_5_loss/categorical_crossentropy/weighted_loss/num_elements/Cast*
T0*S
_classI
GEloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv*
_output_shapes
: 
�
atraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/SumSumetraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/RealDivstraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/BroadcastGradientArgs*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0*S
_classI
GEloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv
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
ctraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/Sum_1Sumatraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/mulutraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*S
_classI
GEloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv*
_output_shapes
: 
�
gtraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/Reshape_1Reshapectraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/Sum_1etraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/Shape_1*
T0*
Tshape0*S
_classI
GEloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv*
_output_shapes
: 
�
gtraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/Sum_grad/Reshape/shapeConst*
valueB:*O
_classE
CAloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/Sum*
dtype0*
_output_shapes
:
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
]training/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul_grad/SumSum]training/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul_grad/Mulotraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*O
_classE
CAloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul*
_output_shapes
:
�
atraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul_grad/ReshapeReshape]training/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul_grad/Sum_training/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul_grad/Shape*
T0*
Tshape0*O
_classE
CAloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul*#
_output_shapes
:���������
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
{training/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape_2_grad/ReshapeReshapectraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul_grad/Reshape_1ytraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape_2_grad/Shape*#
_output_shapes
:���������*
T0*
Tshape0*i
_class_
][loc:@loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape_2
�
"training/Adam/gradients/zeros_like	ZerosLikeNloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits:1*0
_output_shapes
:������������������*
T0*_
_classU
SQloc:@loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits
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
mtraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits_grad/mulMulttraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits_grad/ExpandDimsNloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits:1*
T0*_
_classU
SQloc:@loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits*0
_output_shapes
:������������������
�
ttraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits_grad/LogSoftmax
LogSoftmaxTloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape*
T0*_
_classU
SQloc:@loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits*0
_output_shapes
:������������������
�
mtraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits_grad/NegNegttraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits_grad/LogSoftmax*0
_output_shapes
:������������������*
T0*_
_classU
SQloc:@loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits
�
ztraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits_grad/ExpandDims_1/dimConst*
valueB :
���������*_
_classU
SQloc:@loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits*
dtype0*
_output_shapes
: 
�
vtraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits_grad/ExpandDims_1
ExpandDims{training/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape_2_grad/Reshapeztraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits_grad/ExpandDims_1/dim*

Tdim0*
T0*_
_classU
SQloc:@loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits*'
_output_shapes
:���������
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
ytraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape_grad/ReshapeReshapemtraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits_grad/mulwtraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape_grad/Shape*'
_output_shapes
:���������+*
T0*
Tshape0*g
_class]
[Yloc:@loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape
�
8training/Adam/gradients/dense_5/BiasAdd_grad/BiasAddGradBiasAddGradytraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape_grad/Reshape*
data_formatNHWC*
_output_shapes
:+*
T0*"
_class
loc:@dense_5/BiasAdd
�
2training/Adam/gradients/dense_5/MatMul_grad/MatMulMatMulytraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape_grad/Reshapedense_5/MatMul/ReadVariableOp*(
_output_shapes
:����������*
transpose_a( *
transpose_b(*
T0*!
_class
loc:@dense_5/MatMul
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
Otraining/Adam/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/Shape_1Shape(batch_normalization_4/cond/batchnorm/sub*
_output_shapes
:*
T0*
out_type0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/add_1
�
]training/Adam/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsMtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/ShapeOtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/Shape_1*
T0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/add_1*2
_output_shapes 
:���������:���������
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
Otraining/Adam/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/ReshapeReshapeKtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/SumMtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/Shape*
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/add_1*(
_output_shapes
:����������
�
Mtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/Sum_1SumGtraining/Adam/gradients/batch_normalization_4/cond/Merge_grad/cond_grad_training/Adam/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/add_1
�
Qtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/Reshape_1ReshapeMtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/Sum_1Otraining/Adam/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/Shape_1*
_output_shapes	
:�*
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/add_1
�
training/Adam/gradients/SwitchSwitch%batch_normalization_4/batchnorm/add_1"batch_normalization_4/cond/pred_id*<
_output_shapes*
(:����������:����������*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1
�
 training/Adam/gradients/IdentityIdentitytraining/Adam/gradients/Switch*(
_output_shapes
:����������*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1
�
training/Adam/gradients/Shape_1Shapetraining/Adam/gradients/Switch*
T0*
out_type0*8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1*
_output_shapes
:
�
#training/Adam/gradients/zeros/ConstConst!^training/Adam/gradients/Identity*
dtype0*
_output_shapes
: *
valueB
 *    *8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1
�
training/Adam/gradients/zerosFilltraining/Adam/gradients/Shape_1#training/Adam/gradients/zeros/Const*(
_output_shapes
:����������*
T0*

index_type0*8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1
�
Jtraining/Adam/gradients/batch_normalization_4/cond/Switch_1_grad/cond_gradMergetraining/Adam/gradients/zerosItraining/Adam/gradients/batch_normalization_4/cond/Merge_grad/cond_grad:1*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1*
N**
_output_shapes
:����������: 
�
Mtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/ShapeShape1batch_normalization_4/cond/batchnorm/mul_1/Switch*
_output_shapes
:*
T0*
out_type0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/mul_1
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
Ktraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/SumSumKtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/Mul]training/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/BroadcastGradientArgs*
T0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/mul_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
Otraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/ReshapeReshapeKtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/SumMtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/mul_1*(
_output_shapes
:����������
�
Mtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/Mul_1Mul1batch_normalization_4/cond/batchnorm/mul_1/SwitchOtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/Reshape*(
_output_shapes
:����������*
T0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/mul_1
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
Qtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/Reshape_1ReshapeMtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/Sum_1Otraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/mul_1*
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
:*

Tidx0*
	keep_dims( 
�
Jtraining/Adam/gradients/batch_normalization_4/batchnorm/add_1_grad/ReshapeReshapeFtraining/Adam/gradients/batch_normalization_4/batchnorm/add_1_grad/SumHtraining/Adam/gradients/batch_normalization_4/batchnorm/add_1_grad/Shape*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1*(
_output_shapes
:����������
�
Htraining/Adam/gradients/batch_normalization_4/batchnorm/add_1_grad/Sum_1SumJtraining/Adam/gradients/batch_normalization_4/cond/Switch_1_grad/cond_gradZtraining/Adam/gradients/batch_normalization_4/batchnorm/add_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1*
_output_shapes
:
�
Ltraining/Adam/gradients/batch_normalization_4/batchnorm/add_1_grad/Reshape_1ReshapeHtraining/Adam/gradients/batch_normalization_4/batchnorm/add_1_grad/Sum_1Jtraining/Adam/gradients/batch_normalization_4/batchnorm/add_1_grad/Shape_1*
_output_shapes	
:�*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1
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
Jtraining/Adam/gradients/batch_normalization_4/batchnorm/mul_1_grad/Shape_1Shape#batch_normalization_4/batchnorm/mul*
T0*
out_type0*8
_class.
,*loc:@batch_normalization_4/batchnorm/mul_1*
_output_shapes
:
�
Xtraining/Adam/gradients/batch_normalization_4/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsHtraining/Adam/gradients/batch_normalization_4/batchnorm/mul_1_grad/ShapeJtraining/Adam/gradients/batch_normalization_4/batchnorm/mul_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/mul_1
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
%training/Adam/gradients/VariableShapeVariableShape"training/Adam/gradients/Switch_2:1#^training/Adam/gradients/Identity_2*
out_type0*-
_class#
!loc:@batch_normalization_4/beta*
_output_shapes
:
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
T0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/mul_1*
N*
_output_shapes	
:�
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
Ftraining/Adam/gradients/batch_normalization_4/batchnorm/mul_2_grad/MulMulDtraining/Adam/gradients/batch_normalization_4/batchnorm/sub_grad/Neg#batch_normalization_4/batchnorm/mul*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/mul_2*
_output_shapes	
:�
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
"training/Adam/gradients/Identity_3Identity"training/Adam/gradients/Switch_3:1*
T0*.
_class$
" loc:@batch_normalization_4/gamma*
_output_shapes
: 
�
'training/Adam/gradients/VariableShape_1VariableShape"training/Adam/gradients/Switch_3:1#^training/Adam/gradients/Identity_3*
_output_shapes
:*
out_type0*.
_class$
" loc:@batch_normalization_4/gamma
�
%training/Adam/gradients/zeros_3/ConstConst#^training/Adam/gradients/Identity_3*
dtype0*
_output_shapes
: *
valueB
 *    *.
_class$
" loc:@batch_normalization_4/gamma
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
Ytraining/Adam/gradients/batch_normalization_4/batchnorm/add_grad/BroadcastGradientArgs/s0Const*
dtype0*
_output_shapes
:*
valueB:�*6
_class,
*(loc:@batch_normalization_4/batchnorm/add
�
Ytraining/Adam/gradients/batch_normalization_4/batchnorm/add_grad/BroadcastGradientArgs/s1Const*
valueB *6
_class,
*(loc:@batch_normalization_4/batchnorm/add*
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
Ltraining/Adam/gradients/batch_normalization_4/moments/Squeeze_1_grad/ReshapeReshapeLtraining/Adam/gradients/batch_normalization_4/batchnorm/Rsqrt_grad/RsqrtGradJtraining/Adam/gradients/batch_normalization_4/moments/Squeeze_1_grad/Shape*
_output_shapes
:	�*
T0*
Tshape0*:
_class0
.,loc:@batch_normalization_4/moments/Squeeze_1
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
Ktraining/Adam/gradients/batch_normalization_4/moments/variance_grad/Shape_1Const*
valueB:*9
_class/
-+loc:@batch_normalization_4/moments/variance*
dtype0*
_output_shapes
:
�
Otraining/Adam/gradients/batch_normalization_4/moments/variance_grad/range/startConst*
value	B : *9
_class/
-+loc:@batch_normalization_4/moments/variance*
dtype0*
_output_shapes
: 
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
Htraining/Adam/gradients/batch_normalization_4/moments/variance_grad/FillFillKtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/Shape_1Ntraining/Adam/gradients/batch_normalization_4/moments/variance_grad/Fill/value*
_output_shapes
:*
T0*

index_type0*9
_class/
-+loc:@batch_normalization_4/moments/variance
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
Ktraining/Adam/gradients/batch_normalization_4/moments/variance_grad/ReshapeReshapeLtraining/Adam/gradients/batch_normalization_4/moments/Squeeze_1_grad/ReshapeQtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/DynamicStitch*0
_output_shapes
:������������������*
T0*
Tshape0*9
_class/
-+loc:@batch_normalization_4/moments/variance
�
Htraining/Adam/gradients/batch_normalization_4/moments/variance_grad/TileTileKtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/ReshapeLtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/floordiv*

Tmultiples0*
T0*9
_class/
-+loc:@batch_normalization_4/moments/variance*0
_output_shapes
:������������������
�
Ktraining/Adam/gradients/batch_normalization_4/moments/variance_grad/Shape_2Shape/batch_normalization_4/moments/SquaredDifference*
T0*
out_type0*9
_class/
-+loc:@batch_normalization_4/moments/variance*
_output_shapes
:
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
Jtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/Prod_1ProdKtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/Shape_3Ktraining/Adam/gradients/batch_normalization_4/moments/variance_grad/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0*9
_class/
-+loc:@batch_normalization_4/moments/variance
�
Otraining/Adam/gradients/batch_normalization_4/moments/variance_grad/Maximum_1/yConst*
value	B :*9
_class/
-+loc:@batch_normalization_4/moments/variance*
dtype0*
_output_shapes
: 
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
Htraining/Adam/gradients/batch_normalization_4/moments/variance_grad/CastCastNtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/floordiv_1*

SrcT0*9
_class/
-+loc:@batch_normalization_4/moments/variance*
Truncate( *
_output_shapes
: *

DstT0
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
: *
valueB
 *   @*B
_class8
64loc:@batch_normalization_4/moments/SquaredDifference
�
Ptraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/MulMulStraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/scalarKtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/truediv*(
_output_shapes
:����������*
T0*B
_class8
64loc:@batch_normalization_4/moments/SquaredDifference
�
Ptraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/subSubdense_4/Relu*batch_normalization_4/moments/StopGradientL^training/Adam/gradients/batch_normalization_4/moments/variance_grad/truediv*(
_output_shapes
:����������*
T0*B
_class8
64loc:@batch_normalization_4/moments/SquaredDifference
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
Ttraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/Shape_1Shape*batch_normalization_4/moments/StopGradient*
T0*
out_type0*B
_class8
64loc:@batch_normalization_4/moments/SquaredDifference*
_output_shapes
:
�
btraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgsRtraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/ShapeTtraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0*B
_class8
64loc:@batch_normalization_4/moments/SquaredDifference
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
Rtraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/Sum_1SumRtraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/mul_1dtraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*B
_class8
64loc:@batch_normalization_4/moments/SquaredDifference
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
Gtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:*5
_class+
)'loc:@batch_normalization_4/moments/mean
�
Ktraining/Adam/gradients/batch_normalization_4/moments/mean_grad/range/startConst*
value	B : *5
_class+
)'loc:@batch_normalization_4/moments/mean*
dtype0*
_output_shapes
: 
�
Ktraining/Adam/gradients/batch_normalization_4/moments/mean_grad/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :*5
_class+
)'loc:@batch_normalization_4/moments/mean
�
Etraining/Adam/gradients/batch_normalization_4/moments/mean_grad/rangeRangeKtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/range/startDtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/SizeKtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/range/delta*

Tidx0*5
_class+
)'loc:@batch_normalization_4/moments/mean*
_output_shapes
:
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
T0*
Tshape0*5
_class+
)'loc:@batch_normalization_4/moments/mean
�
Dtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/TileTileGtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/ReshapeHtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/floordiv*
T0*5
_class+
)'loc:@batch_normalization_4/moments/mean*0
_output_shapes
:������������������*

Tmultiples0
�
Gtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Shape_2Shapedense_4/Relu*
T0*
out_type0*5
_class+
)'loc:@batch_normalization_4/moments/mean*
_output_shapes
:
�
Gtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Shape_3Const*
valueB"   �   *5
_class+
)'loc:@batch_normalization_4/moments/mean*
dtype0*
_output_shapes
:
�
Etraining/Adam/gradients/batch_normalization_4/moments/mean_grad/ConstConst*
valueB: *5
_class+
)'loc:@batch_normalization_4/moments/mean*
dtype0*
_output_shapes
:
�
Dtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/ProdProdGtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Shape_2Etraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0*5
_class+
)'loc:@batch_normalization_4/moments/mean
�
Gtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: *5
_class+
)'loc:@batch_normalization_4/moments/mean
�
Ftraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Prod_1ProdGtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Shape_3Gtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Const_1*

Tidx0*
	keep_dims( *
T0*5
_class+
)'loc:@batch_normalization_4/moments/mean*
_output_shapes
: 
�
Ktraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Maximum_1/yConst*
value	B :*5
_class+
)'loc:@batch_normalization_4/moments/mean*
dtype0*
_output_shapes
: 
�
Itraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Maximum_1MaximumFtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Prod_1Ktraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Maximum_1/y*
_output_shapes
: *
T0*5
_class+
)'loc:@batch_normalization_4/moments/mean
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
N*(
_output_shapes
:����������*
T0*
_class
loc:@dense_4/Relu
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
;training/Adam/gradients/dropout_2/cond/Merge_grad/cond_gradSwitch2training/Adam/gradients/dense_4/MatMul_grad/MatMuldropout_2/cond/pred_id*
T0*!
_class
loc:@dense_4/MatMul*<
_output_shapes*
(:����������:����������
�
 training/Adam/gradients/Switch_4Switch batch_normalization_3/cond/Mergedropout_2/cond/pred_id*
T0*3
_class)
'%loc:@batch_normalization_3/cond/Merge*<
_output_shapes*
(:����������:����������
�
"training/Adam/gradients/Identity_4Identity"training/Adam/gradients/Switch_4:1*
T0*3
_class)
'%loc:@batch_normalization_3/cond/Merge*(
_output_shapes
:����������
�
training/Adam/gradients/Shape_3Shape"training/Adam/gradients/Switch_4:1*
_output_shapes
:*
T0*
out_type0*3
_class)
'%loc:@batch_normalization_3/cond/Merge
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
?training/Adam/gradients/dropout_2/cond/dropout/mul_1_grad/ShapeShapedropout_2/cond/dropout/mul*
_output_shapes
:*
T0*
out_type0*/
_class%
#!loc:@dropout_2/cond/dropout/mul_1
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
=training/Adam/gradients/dropout_2/cond/dropout/mul_1_grad/MulMul=training/Adam/gradients/dropout_2/cond/Merge_grad/cond_grad:1dropout_2/cond/dropout/Cast*
T0*/
_class%
#!loc:@dropout_2/cond/dropout/mul_1*(
_output_shapes
:����������
�
=training/Adam/gradients/dropout_2/cond/dropout/mul_1_grad/SumSum=training/Adam/gradients/dropout_2/cond/dropout/mul_1_grad/MulOtraining/Adam/gradients/dropout_2/cond/dropout/mul_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*/
_class%
#!loc:@dropout_2/cond/dropout/mul_1*
_output_shapes
:
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
?training/Adam/gradients/dropout_2/cond/dropout/mul_1_grad/Sum_1Sum?training/Adam/gradients/dropout_2/cond/dropout/mul_1_grad/Mul_1Qtraining/Adam/gradients/dropout_2/cond/dropout/mul_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*/
_class%
#!loc:@dropout_2/cond/dropout/mul_1*
_output_shapes
:
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
=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Mul_1Mul%dropout_2/cond/dropout/Shape/Switch:1Atraining/Adam/gradients/dropout_2/cond/dropout/mul_1_grad/Reshape*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul*(
_output_shapes
:����������
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
Atraining/Adam/gradients/dropout_2/cond/dropout/mul_grad/Reshape_1Reshape=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Sum_1?training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Shape_1*
T0*
Tshape0*-
_class#
!loc:@dropout_2/cond/dropout/mul*
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
"training/Adam/gradients/Identity_5Identity training/Adam/gradients/Switch_5*
T0*3
_class)
'%loc:@batch_normalization_3/cond/Merge*(
_output_shapes
:����������
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
training/Adam/gradients/zeros_5Filltraining/Adam/gradients/Shape_4%training/Adam/gradients/zeros_5/Const*
T0*

index_type0*3
_class)
'%loc:@batch_normalization_3/cond/Merge*(
_output_shapes
:����������
�
Jtraining/Adam/gradients/dropout_2/cond/dropout/Shape/Switch_grad/cond_gradMergetraining/Adam/gradients/zeros_5?training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Reshape*
N**
_output_shapes
:����������: *
T0*3
_class)
'%loc:@batch_normalization_3/cond/Merge
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
]training/Adam/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsMtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/ShapeOtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/Shape_1*
T0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/add_1*2
_output_shapes 
:���������:���������
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
Otraining/Adam/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/ReshapeReshapeKtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/SumMtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/add_1
�
Mtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/Sum_1SumGtraining/Adam/gradients/batch_normalization_3/cond/Merge_grad/cond_grad_training/Adam/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/add_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
Qtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/Reshape_1ReshapeMtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/Sum_1Otraining/Adam/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/Shape_1*
_output_shapes	
:�*
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/add_1
�
 training/Adam/gradients/Switch_6Switch%batch_normalization_3/batchnorm/add_1"batch_normalization_3/cond/pred_id*<
_output_shapes*
(:����������:����������*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1
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
%training/Adam/gradients/zeros_6/ConstConst#^training/Adam/gradients/Identity_6*
dtype0*
_output_shapes
: *
valueB
 *    *8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1
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
Mtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/ShapeShape1batch_normalization_3/cond/batchnorm/mul_1/Switch*
_output_shapes
:*
T0*
out_type0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/mul_1
�
Otraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/Shape_1Shape(batch_normalization_3/cond/batchnorm/mul*
T0*
out_type0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/mul_1*
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
Ktraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/MulMulOtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/Reshape(batch_normalization_3/cond/batchnorm/mul*(
_output_shapes
:����������*
T0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/mul_1
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
Otraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/ReshapeReshapeKtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/SumMtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/mul_1*(
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
:*

Tidx0*
	keep_dims( *
T0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/mul_1
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
Jtraining/Adam/gradients/batch_normalization_3/batchnorm/add_1_grad/Shape_1Shape#batch_normalization_3/batchnorm/sub*
_output_shapes
:*
T0*
out_type0*8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1
�
Xtraining/Adam/gradients/batch_normalization_3/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsHtraining/Adam/gradients/batch_normalization_3/batchnorm/add_1_grad/ShapeJtraining/Adam/gradients/batch_normalization_3/batchnorm/add_1_grad/Shape_1*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1*2
_output_shapes 
:���������:���������
�
Ftraining/Adam/gradients/batch_normalization_3/batchnorm/add_1_grad/SumSumJtraining/Adam/gradients/batch_normalization_3/cond/Switch_1_grad/cond_gradXtraining/Adam/gradients/batch_normalization_3/batchnorm/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1
�
Jtraining/Adam/gradients/batch_normalization_3/batchnorm/add_1_grad/ReshapeReshapeFtraining/Adam/gradients/batch_normalization_3/batchnorm/add_1_grad/SumHtraining/Adam/gradients/batch_normalization_3/batchnorm/add_1_grad/Shape*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1*(
_output_shapes
:����������
�
Htraining/Adam/gradients/batch_normalization_3/batchnorm/add_1_grad/Sum_1SumJtraining/Adam/gradients/batch_normalization_3/cond/Switch_1_grad/cond_gradZtraining/Adam/gradients/batch_normalization_3/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
Ltraining/Adam/gradients/batch_normalization_3/batchnorm/add_1_grad/Reshape_1ReshapeHtraining/Adam/gradients/batch_normalization_3/batchnorm/add_1_grad/Sum_1Jtraining/Adam/gradients/batch_normalization_3/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1*
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
T0*
out_type0*
_class
loc:@dense_3/Relu*
_output_shapes
:
�
%training/Adam/gradients/zeros_7/ConstConst#^training/Adam/gradients/Identity_7*
dtype0*
_output_shapes
: *
valueB
 *    *
_class
loc:@dense_3/Relu
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
N**
_output_shapes
:����������: *
T0*
_class
loc:@dense_3/Relu
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
Htraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/ShapeShapedense_3/Relu*
_output_shapes
:*
T0*
out_type0*8
_class.
,*loc:@batch_normalization_3/batchnorm/mul_1
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
Ftraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/MulMulJtraining/Adam/gradients/batch_normalization_3/batchnorm/add_1_grad/Reshape#batch_normalization_3/batchnorm/mul*(
_output_shapes
:����������*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/mul_1
�
Ftraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/SumSumFtraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/MulXtraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/mul_1
�
Jtraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/ReshapeReshapeFtraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/SumHtraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_3/batchnorm/mul_1
�
Htraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/Mul_1Muldense_3/ReluJtraining/Adam/gradients/batch_normalization_3/batchnorm/add_1_grad/Reshape*(
_output_shapes
:����������*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/mul_1
�
Htraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/Sum_1SumHtraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/Mul_1Ztraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/mul_1*
_output_shapes
:
�
Ltraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/Reshape_1ReshapeHtraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/Sum_1Jtraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/Shape_1*
_output_shapes	
:�*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_3/batchnorm/mul_1
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
Htraining/Adam/gradients/batch_normalization_3/moments/Squeeze_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"   �   *8
_class.
,*loc:@batch_normalization_3/moments/Squeeze
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
Ftraining/Adam/gradients/batch_normalization_3/batchnorm/mul_grad/Mul_1Multraining/Adam/gradients/AddN_8%batch_normalization_3/batchnorm/Rsqrt*
T0*6
_class,
*(loc:@batch_normalization_3/batchnorm/mul*
_output_shapes	
:�
�
 training/Adam/gradients/Switch_9Switchbatch_normalization_3/gamma"batch_normalization_3/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_3/gamma*
_output_shapes
: : 
�
"training/Adam/gradients/Identity_9Identity"training/Adam/gradients/Switch_9:1*
_output_shapes
: *
T0*.
_class$
" loc:@batch_normalization_3/gamma
�
'training/Adam/gradients/VariableShape_3VariableShape"training/Adam/gradients/Switch_9:1#^training/Adam/gradients/Identity_9*
_output_shapes
:*
out_type0*.
_class$
" loc:@batch_normalization_3/gamma
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
training/Adam/gradients/zeros_9Fill'training/Adam/gradients/VariableShape_3%training/Adam/gradients/zeros_9/Const*#
_output_shapes
:���������*
T0*

index_type0*.
_class$
" loc:@batch_normalization_3/gamma
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
Dtraining/Adam/gradients/batch_normalization_3/batchnorm/add_grad/SumSumLtraining/Adam/gradients/batch_normalization_3/batchnorm/Rsqrt_grad/RsqrtGradVtraining/Adam/gradients/batch_normalization_3/batchnorm/add_grad/Sum/reduction_indices*

Tidx0*
	keep_dims( *
T0*6
_class,
*(loc:@batch_normalization_3/batchnorm/add*
_output_shapes
: 
�
Ntraining/Adam/gradients/batch_normalization_3/batchnorm/add_grad/Reshape/shapeConst*
dtype0*
_output_shapes
: *
valueB *6
_class,
*(loc:@batch_normalization_3/batchnorm/add
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
dtype0*
_output_shapes
:*
valueB"   �   *:
_class0
.,loc:@batch_normalization_3/moments/Squeeze_1
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
Htraining/Adam/gradients/batch_normalization_3/moments/variance_grad/SizeConst*
dtype0*
_output_shapes
: *
value	B :*9
_class/
-+loc:@batch_normalization_3/moments/variance
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
Otraining/Adam/gradients/batch_normalization_3/moments/variance_grad/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :*9
_class/
-+loc:@batch_normalization_3/moments/variance
�
Itraining/Adam/gradients/batch_normalization_3/moments/variance_grad/rangeRangeOtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/range/startHtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/SizeOtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/range/delta*
_output_shapes
:*

Tidx0*9
_class/
-+loc:@batch_normalization_3/moments/variance
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
Ktraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: *9
_class/
-+loc:@batch_normalization_3/moments/variance
�
Jtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Prod_1ProdKtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Shape_3Ktraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Const_1*
T0*9
_class/
-+loc:@batch_normalization_3/moments/variance*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
Otraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Maximum_1/yConst*
value	B :*9
_class/
-+loc:@batch_normalization_3/moments/variance*
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
Truncate( *
_output_shapes
: *

DstT0
�
Ktraining/Adam/gradients/batch_normalization_3/moments/variance_grad/truedivRealDivHtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/TileHtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Cast*
T0*9
_class/
-+loc:@batch_normalization_3/moments/variance*(
_output_shapes
:����������
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
Ptraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/subSubdense_3/Relu*batch_normalization_3/moments/StopGradientL^training/Adam/gradients/batch_normalization_3/moments/variance_grad/truediv*(
_output_shapes
:����������*
T0*B
_class8
64loc:@batch_normalization_3/moments/SquaredDifference
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
Rtraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/Sum_1SumRtraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/mul_1dtraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*B
_class8
64loc:@batch_normalization_3/moments/SquaredDifference
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
Dtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/SizeConst*
dtype0*
_output_shapes
: *
value	B :*5
_class+
)'loc:@batch_normalization_3/moments/mean
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
Ktraining/Adam/gradients/batch_normalization_3/moments/mean_grad/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :*5
_class+
)'loc:@batch_normalization_3/moments/mean
�
Etraining/Adam/gradients/batch_normalization_3/moments/mean_grad/rangeRangeKtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/range/startDtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/SizeKtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/range/delta*
_output_shapes
:*

Tidx0*5
_class+
)'loc:@batch_normalization_3/moments/mean
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
Gtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/MaximumMaximumMtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/DynamicStitchItraining/Adam/gradients/batch_normalization_3/moments/mean_grad/Maximum/y*
_output_shapes
:*
T0*5
_class+
)'loc:@batch_normalization_3/moments/mean
�
Htraining/Adam/gradients/batch_normalization_3/moments/mean_grad/floordivFloorDivEtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/ShapeGtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/Maximum*
_output_shapes
:*
T0*5
_class+
)'loc:@batch_normalization_3/moments/mean
�
Gtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/ReshapeReshapeJtraining/Adam/gradients/batch_normalization_3/moments/Squeeze_grad/ReshapeMtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/DynamicStitch*
T0*
Tshape0*5
_class+
)'loc:@batch_normalization_3/moments/mean*0
_output_shapes
:������������������
�
Dtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/TileTileGtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/ReshapeHtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/floordiv*0
_output_shapes
:������������������*

Tmultiples0*
T0*5
_class+
)'loc:@batch_normalization_3/moments/mean
�
Gtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/Shape_2Shapedense_3/Relu*
T0*
out_type0*5
_class+
)'loc:@batch_normalization_3/moments/mean*
_output_shapes
:
�
Gtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/Shape_3Const*
valueB"   �   *5
_class+
)'loc:@batch_normalization_3/moments/mean*
dtype0*
_output_shapes
:
�
Etraining/Adam/gradients/batch_normalization_3/moments/mean_grad/ConstConst*
valueB: *5
_class+
)'loc:@batch_normalization_3/moments/mean*
dtype0*
_output_shapes
:
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
Ftraining/Adam/gradients/batch_normalization_3/moments/mean_grad/Prod_1ProdGtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/Shape_3Gtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/Const_1*

Tidx0*
	keep_dims( *
T0*5
_class+
)'loc:@batch_normalization_3/moments/mean*
_output_shapes
: 
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
2training/Adam/gradients/dense_3/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_3/Relu_grad/ReluGraddense_3/MatMul/ReadVariableOp*
T0*!
_class
loc:@dense_3/MatMul*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
4training/Adam/gradients/dense_3/MatMul_grad/MatMul_1MatMuldropout_1/cond/Merge2training/Adam/gradients/dense_3/Relu_grad/ReluGrad* 
_output_shapes
:
��*
transpose_a(*
transpose_b( *
T0*!
_class
loc:@dense_3/MatMul
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
training/Adam/gradients/Shape_7Shape#training/Adam/gradients/Switch_10:1*
T0*
out_type0*3
_class)
'%loc:@batch_normalization_2/cond/Merge*
_output_shapes
:
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
 training/Adam/gradients/zeros_10Filltraining/Adam/gradients/Shape_7&training/Adam/gradients/zeros_10/Const*
T0*

index_type0*3
_class)
'%loc:@batch_normalization_2/cond/Merge*(
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
?training/Adam/gradients/dropout_1/cond/dropout/mul_1_grad/ShapeShapedropout_1/cond/dropout/mul*
T0*
out_type0*/
_class%
#!loc:@dropout_1/cond/dropout/mul_1*
_output_shapes
:
�
Atraining/Adam/gradients/dropout_1/cond/dropout/mul_1_grad/Shape_1Shapedropout_1/cond/dropout/Cast*
T0*
out_type0*/
_class%
#!loc:@dropout_1/cond/dropout/mul_1*
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
=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/ShapeShape%dropout_1/cond/dropout/Shape/Switch:1*
T0*
out_type0*-
_class#
!loc:@dropout_1/cond/dropout/mul*
_output_shapes
:
�
?training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Shape_1Shapedropout_1/cond/dropout/truediv*
T0*
out_type0*-
_class#
!loc:@dropout_1/cond/dropout/mul*
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
;training/Adam/gradients/dropout_1/cond/dropout/mul_grad/MulMulAtraining/Adam/gradients/dropout_1/cond/dropout/mul_1_grad/Reshapedropout_1/cond/dropout/truediv*(
_output_shapes
:����������*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul
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
!training/Adam/gradients/Switch_11Switch batch_normalization_2/cond/Mergedropout_1/cond/pred_id*<
_output_shapes*
(:����������:����������*
T0*3
_class)
'%loc:@batch_normalization_2/cond/Merge
�
#training/Adam/gradients/Identity_11Identity!training/Adam/gradients/Switch_11*(
_output_shapes
:����������*
T0*3
_class)
'%loc:@batch_normalization_2/cond/Merge
�
training/Adam/gradients/Shape_8Shape!training/Adam/gradients/Switch_11*
_output_shapes
:*
T0*
out_type0*3
_class)
'%loc:@batch_normalization_2/cond/Merge
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
 training/Adam/gradients/zeros_11Filltraining/Adam/gradients/Shape_8&training/Adam/gradients/zeros_11/Const*(
_output_shapes
:����������*
T0*

index_type0*3
_class)
'%loc:@batch_normalization_2/cond/Merge
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
Gtraining/Adam/gradients/batch_normalization_2/cond/Merge_grad/cond_gradSwitchtraining/Adam/gradients/AddN_11"batch_normalization_2/cond/pred_id*<
_output_shapes*
(:����������:����������*
T0*3
_class)
'%loc:@batch_normalization_2/cond/Merge
�
Mtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/ShapeShape*batch_normalization_2/cond/batchnorm/mul_1*
_output_shapes
:*
T0*
out_type0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/add_1
�
Otraining/Adam/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/Shape_1Shape(batch_normalization_2/cond/batchnorm/sub*
T0*
out_type0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/add_1*
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
:*

Tidx0*
	keep_dims( 
�
Otraining/Adam/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/ReshapeReshapeKtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/SumMtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/Shape*
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/add_1*(
_output_shapes
:����������
�
Mtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/Sum_1SumGtraining/Adam/gradients/batch_normalization_2/cond/Merge_grad/cond_grad_training/Adam/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/add_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
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
 training/Adam/gradients/zeros_12Filltraining/Adam/gradients/Shape_9&training/Adam/gradients/zeros_12/Const*(
_output_shapes
:����������*
T0*

index_type0*8
_class.
,*loc:@batch_normalization_2/batchnorm/add_1
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
Otraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/Shape_1Shape(batch_normalization_2/cond/batchnorm/mul*
T0*
out_type0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/mul_1*
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
Ktraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/SumSumKtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/Mul]training/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/mul_1*
_output_shapes
:
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
Mtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/Sum_1SumMtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/Mul_1_training/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
T0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/mul_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
Qtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/Reshape_1ReshapeMtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/Sum_1Otraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/Shape_1*
_output_shapes	
:�*
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/mul_1
�
Itraining/Adam/gradients/batch_normalization_2/cond/batchnorm/sub_grad/NegNegQtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/Reshape_1*
T0*;
_class1
/-loc:@batch_normalization_2/cond/batchnorm/sub*
_output_shapes	
:�
�
Htraining/Adam/gradients/batch_normalization_2/batchnorm/add_1_grad/ShapeShape%batch_normalization_2/batchnorm/mul_1*
T0*
out_type0*8
_class.
,*loc:@batch_normalization_2/batchnorm/add_1*
_output_shapes
:
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
Jtraining/Adam/gradients/batch_normalization_2/batchnorm/add_1_grad/ReshapeReshapeFtraining/Adam/gradients/batch_normalization_2/batchnorm/add_1_grad/SumHtraining/Adam/gradients/batch_normalization_2/batchnorm/add_1_grad/Shape*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_2/batchnorm/add_1*(
_output_shapes
:����������
�
Htraining/Adam/gradients/batch_normalization_2/batchnorm/add_1_grad/Sum_1SumJtraining/Adam/gradients/batch_normalization_2/cond/Switch_1_grad/cond_gradZtraining/Adam/gradients/batch_normalization_2/batchnorm/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/add_1
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
&training/Adam/gradients/zeros_13/ConstConst$^training/Adam/gradients/Identity_13*
dtype0*
_output_shapes
: *
valueB
 *    *
_class
loc:@dense_2/Relu
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
Ftraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/SumSumFtraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/MulXtraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/mul_1
�
Jtraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/ReshapeReshapeFtraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/SumHtraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_2/batchnorm/mul_1
�
Htraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/Mul_1Muldense_2/ReluJtraining/Adam/gradients/batch_normalization_2/batchnorm/add_1_grad/Reshape*(
_output_shapes
:����������*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/mul_1
�
Htraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/Sum_1SumHtraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/Mul_1Ztraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/mul_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
Ltraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/Reshape_1ReshapeHtraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/Sum_1Jtraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_2/batchnorm/mul_1*
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
Itraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_grad/MulMultraining/Adam/gradients/AddN_127batch_normalization_2/cond/batchnorm/mul/ReadVariableOp*
_output_shapes	
:�*
T0*;
_class1
/-loc:@batch_normalization_2/cond/batchnorm/mul
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
!training/Adam/gradients/Switch_15Switchbatch_normalization_2/gamma"batch_normalization_2/cond/pred_id*
_output_shapes
: : *
T0*.
_class$
" loc:@batch_normalization_2/gamma
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
 training/Adam/gradients/zeros_15Fill'training/Adam/gradients/VariableShape_5&training/Adam/gradients/zeros_15/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization_2/gamma*#
_output_shapes
:���������
�
etraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul/ReadVariableOp/Switch_grad/cond_gradMergeKtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_grad/Mul_1 training/Adam/gradients/zeros_15*
N*%
_output_shapes
:���������: *
T0*.
_class$
" loc:@batch_normalization_2/gamma
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
Dtraining/Adam/gradients/batch_normalization_2/batchnorm/add_grad/SumSumLtraining/Adam/gradients/batch_normalization_2/batchnorm/Rsqrt_grad/RsqrtGradVtraining/Adam/gradients/batch_normalization_2/batchnorm/add_grad/Sum/reduction_indices*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0*6
_class,
*(loc:@batch_normalization_2/batchnorm/add
�
Ntraining/Adam/gradients/batch_normalization_2/batchnorm/add_grad/Reshape/shapeConst*
dtype0*
_output_shapes
: *
valueB *6
_class,
*(loc:@batch_normalization_2/batchnorm/add
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
N*
_output_shapes	
:�*
T0*.
_class$
" loc:@batch_normalization_2/gamma
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
Ktraining/Adam/gradients/batch_normalization_2/moments/variance_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:*9
_class/
-+loc:@batch_normalization_2/moments/variance
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
Mtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/Maximum/yConst*
value	B :*9
_class/
-+loc:@batch_normalization_2/moments/variance*
dtype0*
_output_shapes
: 
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
Htraining/Adam/gradients/batch_normalization_2/moments/variance_grad/TileTileKtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/ReshapeLtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/floordiv*0
_output_shapes
:������������������*

Tmultiples0*
T0*9
_class/
-+loc:@batch_normalization_2/moments/variance
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
dtype0*
_output_shapes
:*
valueB"   �   *9
_class/
-+loc:@batch_normalization_2/moments/variance
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
Ktraining/Adam/gradients/batch_normalization_2/moments/variance_grad/Const_1Const*
valueB: *9
_class/
-+loc:@batch_normalization_2/moments/variance*
dtype0*
_output_shapes
:
�
Jtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/Prod_1ProdKtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/Shape_3Ktraining/Adam/gradients/batch_normalization_2/moments/variance_grad/Const_1*

Tidx0*
	keep_dims( *
T0*9
_class/
-+loc:@batch_normalization_2/moments/variance*
_output_shapes
: 
�
Otraining/Adam/gradients/batch_normalization_2/moments/variance_grad/Maximum_1/yConst*
dtype0*
_output_shapes
: *
value	B :*9
_class/
-+loc:@batch_normalization_2/moments/variance
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
Truncate( *
_output_shapes
: *

DstT0
�
Ktraining/Adam/gradients/batch_normalization_2/moments/variance_grad/truedivRealDivHtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/TileHtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/Cast*(
_output_shapes
:����������*
T0*9
_class/
-+loc:@batch_normalization_2/moments/variance
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
Ptraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/SumSumRtraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/mul_1btraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/BroadcastGradientArgs*
T0*B
_class8
64loc:@batch_normalization_2/moments/SquaredDifference*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
Ttraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/ReshapeReshapePtraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/SumRtraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/Shape*
T0*
Tshape0*B
_class8
64loc:@batch_normalization_2/moments/SquaredDifference*(
_output_shapes
:����������
�
Rtraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/Sum_1SumRtraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/mul_1dtraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*B
_class8
64loc:@batch_normalization_2/moments/SquaredDifference
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
Ctraining/Adam/gradients/batch_normalization_2/moments/mean_grad/addAddV24batch_normalization_2/moments/mean/reduction_indicesDtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Size*
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean*
_output_shapes
:
�
Ctraining/Adam/gradients/batch_normalization_2/moments/mean_grad/modFloorModCtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/addDtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Size*
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean*
_output_shapes
:
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
Etraining/Adam/gradients/batch_normalization_2/moments/mean_grad/rangeRangeKtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/range/startDtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/SizeKtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/range/delta*5
_class+
)'loc:@batch_normalization_2/moments/mean*
_output_shapes
:*

Tidx0
�
Jtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Fill/valueConst*
value	B :*5
_class+
)'loc:@batch_normalization_2/moments/mean*
dtype0*
_output_shapes
: 
�
Dtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/FillFillGtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Shape_1Jtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Fill/value*
T0*

index_type0*5
_class+
)'loc:@batch_normalization_2/moments/mean*
_output_shapes
:
�
Mtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/DynamicStitchDynamicStitchEtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/rangeCtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/modEtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/ShapeDtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Fill*
N*
_output_shapes
:*
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean
�
Itraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :*5
_class+
)'loc:@batch_normalization_2/moments/mean
�
Gtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/MaximumMaximumMtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/DynamicStitchItraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Maximum/y*
_output_shapes
:*
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean
�
Htraining/Adam/gradients/batch_normalization_2/moments/mean_grad/floordivFloorDivEtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/ShapeGtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Maximum*
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean*
_output_shapes
:
�
Gtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/ReshapeReshapeJtraining/Adam/gradients/batch_normalization_2/moments/Squeeze_grad/ReshapeMtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/DynamicStitch*
T0*
Tshape0*5
_class+
)'loc:@batch_normalization_2/moments/mean*0
_output_shapes
:������������������
�
Dtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/TileTileGtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/ReshapeHtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/floordiv*
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean*0
_output_shapes
:������������������*

Tmultiples0
�
Gtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Shape_2Shapedense_2/Relu*
T0*
out_type0*5
_class+
)'loc:@batch_normalization_2/moments/mean*
_output_shapes
:
�
Gtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Shape_3Const*
valueB"   �   *5
_class+
)'loc:@batch_normalization_2/moments/mean*
dtype0*
_output_shapes
:
�
Etraining/Adam/gradients/batch_normalization_2/moments/mean_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: *5
_class+
)'loc:@batch_normalization_2/moments/mean
�
Dtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/ProdProdGtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Shape_2Etraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Const*
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
Gtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: *5
_class+
)'loc:@batch_normalization_2/moments/mean
�
Ftraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Prod_1ProdGtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Shape_3Gtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean
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
Truncate( *
_output_shapes
: *

DstT0
�
Gtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/truedivRealDivDtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/TileDtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Cast*
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean*(
_output_shapes
:����������
�
training/Adam/gradients/AddN_16AddNXtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1/Switch_grad/cond_gradJtraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/ReshapeTtraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/ReshapeGtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/truediv*
T0*
_class
loc:@dense_2/Relu*
N*(
_output_shapes
:����������
�
2training/Adam/gradients/dense_2/Relu_grad/ReluGradReluGradtraining/Adam/gradients/AddN_16dense_2/Relu*
T0*
_class
loc:@dense_2/Relu*(
_output_shapes
:����������
�
8training/Adam/gradients/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad2training/Adam/gradients/dense_2/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:�*
T0*"
_class
loc:@dense_2/BiasAdd
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
Ktraining/Adam/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/SumSumGtraining/Adam/gradients/batch_normalization_1/cond/Merge_grad/cond_grad]training/Adam/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/add_1*
_output_shapes
:
�
Otraining/Adam/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/ReshapeReshapeKtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/SumMtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/add_1
�
Mtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/Sum_1SumGtraining/Adam/gradients/batch_normalization_1/cond/Merge_grad/cond_grad_training/Adam/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/add_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
Qtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/Reshape_1ReshapeMtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/Sum_1Otraining/Adam/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/add_1*
_output_shapes	
:�
�
!training/Adam/gradients/Switch_16Switch%batch_normalization_1/batchnorm/add_1"batch_normalization_1/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1*<
_output_shapes*
(:����������:����������
�
#training/Adam/gradients/Identity_16Identity!training/Adam/gradients/Switch_16*(
_output_shapes
:����������*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1
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
 training/Adam/gradients/zeros_16Fill training/Adam/gradients/Shape_11&training/Adam/gradients/zeros_16/Const*(
_output_shapes
:����������*
T0*

index_type0*8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1
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
T0*
out_type0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/mul_1
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
Ktraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/SumSumKtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/Mul]training/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/BroadcastGradientArgs*
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/mul_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
Otraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/ReshapeReshapeKtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/SumMtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/mul_1
�
Mtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/Mul_1Mul1batch_normalization_1/cond/batchnorm/mul_1/SwitchOtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/Reshape*(
_output_shapes
:����������*
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/mul_1
�
Mtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/Sum_1SumMtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/Mul_1_training/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/mul_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
Qtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/Reshape_1ReshapeMtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/Sum_1Otraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/mul_1*
_output_shapes	
:�
�
Itraining/Adam/gradients/batch_normalization_1/cond/batchnorm/sub_grad/NegNegQtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/Reshape_1*
T0*;
_class1
/-loc:@batch_normalization_1/cond/batchnorm/sub*
_output_shapes	
:�
�
Htraining/Adam/gradients/batch_normalization_1/batchnorm/add_1_grad/ShapeShape%batch_normalization_1/batchnorm/mul_1*
T0*
out_type0*8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1*
_output_shapes
:
�
Jtraining/Adam/gradients/batch_normalization_1/batchnorm/add_1_grad/Shape_1Shape#batch_normalization_1/batchnorm/sub*
T0*
out_type0*8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1*
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
Ftraining/Adam/gradients/batch_normalization_1/batchnorm/add_1_grad/SumSumJtraining/Adam/gradients/batch_normalization_1/cond/Switch_1_grad/cond_gradXtraining/Adam/gradients/batch_normalization_1/batchnorm/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1*
_output_shapes
:
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
#training/Adam/gradients/Identity_17Identity#training/Adam/gradients/Switch_17:1*
T0*
_class
loc:@dense_1/Relu*(
_output_shapes
:����������
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
Ktraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_2_grad/MulMulItraining/Adam/gradients/batch_normalization_1/cond/batchnorm/sub_grad/Neg(batch_normalization_1/cond/batchnorm/mul*
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/mul_2*
_output_shapes	
:�
�
Mtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_2_grad/Mul_1MulItraining/Adam/gradients/batch_normalization_1/cond/batchnorm/sub_grad/Neg5batch_normalization_1/cond/batchnorm/ReadVariableOp_1*
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/mul_2*
_output_shapes	
:�
�
Htraining/Adam/gradients/batch_normalization_1/batchnorm/mul_1_grad/ShapeShapedense_1/Relu*
T0*
out_type0*8
_class.
,*loc:@batch_normalization_1/batchnorm/mul_1*
_output_shapes
:
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
Ftraining/Adam/gradients/batch_normalization_1/batchnorm/mul_1_grad/SumSumFtraining/Adam/gradients/batch_normalization_1/batchnorm/mul_1_grad/MulXtraining/Adam/gradients/batch_normalization_1/batchnorm/mul_1_grad/BroadcastGradientArgs*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/mul_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
Jtraining/Adam/gradients/batch_normalization_1/batchnorm/mul_1_grad/ReshapeReshapeFtraining/Adam/gradients/batch_normalization_1/batchnorm/mul_1_grad/SumHtraining/Adam/gradients/batch_normalization_1/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_1/batchnorm/mul_1*(
_output_shapes
:����������
�
Htraining/Adam/gradients/batch_normalization_1/batchnorm/mul_1_grad/Mul_1Muldense_1/ReluJtraining/Adam/gradients/batch_normalization_1/batchnorm/add_1_grad/Reshape*(
_output_shapes
:����������*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/mul_1
�
Htraining/Adam/gradients/batch_normalization_1/batchnorm/mul_1_grad/Sum_1SumHtraining/Adam/gradients/batch_normalization_1/batchnorm/mul_1_grad/Mul_1Ztraining/Adam/gradients/batch_normalization_1/batchnorm/mul_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/mul_1*
_output_shapes
:
�
Ltraining/Adam/gradients/batch_normalization_1/batchnorm/mul_1_grad/Reshape_1ReshapeHtraining/Adam/gradients/batch_normalization_1/batchnorm/mul_1_grad/Sum_1Jtraining/Adam/gradients/batch_normalization_1/batchnorm/mul_1_grad/Shape_1*
_output_shapes	
:�*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_1/batchnorm/mul_1
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
#training/Adam/gradients/Identity_18Identity#training/Adam/gradients/Switch_18:1*
_output_shapes
: *
T0*-
_class#
!loc:@batch_normalization_1/beta
�
'training/Adam/gradients/VariableShape_6VariableShape#training/Adam/gradients/Switch_18:1$^training/Adam/gradients/Identity_18*
_output_shapes
:*
out_type0*-
_class#
!loc:@batch_normalization_1/beta
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
 training/Adam/gradients/zeros_18Fill'training/Adam/gradients/VariableShape_6&training/Adam/gradients/zeros_18/Const*#
_output_shapes
:���������*
T0*

index_type0*-
_class#
!loc:@batch_normalization_1/beta
�
ctraining/Adam/gradients/batch_normalization_1/cond/batchnorm/ReadVariableOp_2/Switch_grad/cond_gradMergeQtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/Reshape_1 training/Adam/gradients/zeros_18*
N*%
_output_shapes
:���������: *
T0*-
_class#
!loc:@batch_normalization_1/beta
�
training/Adam/gradients/AddN_17AddNQtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/Reshape_1Mtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_2_grad/Mul_1*
N*
_output_shapes	
:�*
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/mul_1
�
Itraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_grad/MulMultraining/Adam/gradients/AddN_177batch_normalization_1/cond/batchnorm/mul/ReadVariableOp*
T0*;
_class1
/-loc:@batch_normalization_1/cond/batchnorm/mul*
_output_shapes	
:�
�
Ktraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_grad/Mul_1Multraining/Adam/gradients/AddN_17*batch_normalization_1/cond/batchnorm/Rsqrt*
T0*;
_class1
/-loc:@batch_normalization_1/cond/batchnorm/mul*
_output_shapes	
:�
�
Ftraining/Adam/gradients/batch_normalization_1/batchnorm/mul_2_grad/MulMulDtraining/Adam/gradients/batch_normalization_1/batchnorm/sub_grad/Neg#batch_normalization_1/batchnorm/mul*
_output_shapes	
:�*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/mul_2
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
Htraining/Adam/gradients/batch_normalization_1/moments/Squeeze_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"   �   *8
_class.
,*loc:@batch_normalization_1/moments/Squeeze
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
!training/Adam/gradients/Switch_19Switchbatch_normalization_1/gamma"batch_normalization_1/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_1/gamma*
_output_shapes
: : 
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
&training/Adam/gradients/zeros_19/ConstConst$^training/Adam/gradients/Identity_19*
valueB
 *    *.
_class$
" loc:@batch_normalization_1/gamma*
dtype0*
_output_shapes
: 
�
 training/Adam/gradients/zeros_19Fill'training/Adam/gradients/VariableShape_7&training/Adam/gradients/zeros_19/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization_1/gamma*#
_output_shapes
:���������
�
etraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul/ReadVariableOp/Switch_grad/cond_gradMergeKtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_grad/Mul_1 training/Adam/gradients/zeros_19*
N*%
_output_shapes
:���������: *
T0*.
_class$
" loc:@batch_normalization_1/gamma
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
Dtraining/Adam/gradients/batch_normalization_1/batchnorm/add_grad/SumSumLtraining/Adam/gradients/batch_normalization_1/batchnorm/Rsqrt_grad/RsqrtGradVtraining/Adam/gradients/batch_normalization_1/batchnorm/add_grad/Sum/reduction_indices*
T0*6
_class,
*(loc:@batch_normalization_1/batchnorm/add*
_output_shapes
: *

Tidx0*
	keep_dims( 
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
T0*.
_class$
" loc:@batch_normalization_1/gamma*
N*
_output_shapes	
:�
�
Jtraining/Adam/gradients/batch_normalization_1/moments/Squeeze_1_grad/ShapeConst*
valueB"   �   *:
_class0
.,loc:@batch_normalization_1/moments/Squeeze_1*
dtype0*
_output_shapes
:
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
Htraining/Adam/gradients/batch_normalization_1/moments/variance_grad/SizeConst*
dtype0*
_output_shapes
: *
value	B :*9
_class/
-+loc:@batch_normalization_1/moments/variance
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
Mtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Maximum/yConst*
value	B :*9
_class/
-+loc:@batch_normalization_1/moments/variance*
dtype0*
_output_shapes
: 
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
Ktraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Shape_2Shape/batch_normalization_1/moments/SquaredDifference*
T0*
out_type0*9
_class/
-+loc:@batch_normalization_1/moments/variance*
_output_shapes
:
�
Ktraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Shape_3Const*
dtype0*
_output_shapes
:*
valueB"   �   *9
_class/
-+loc:@batch_normalization_1/moments/variance
�
Itraining/Adam/gradients/batch_normalization_1/moments/variance_grad/ConstConst*
valueB: *9
_class/
-+loc:@batch_normalization_1/moments/variance*
dtype0*
_output_shapes
:
�
Htraining/Adam/gradients/batch_normalization_1/moments/variance_grad/ProdProdKtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Shape_2Itraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Const*
T0*9
_class/
-+loc:@batch_normalization_1/moments/variance*
_output_shapes
: *

Tidx0*
	keep_dims( 
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
Mtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Maximum_1MaximumJtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Prod_1Otraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Maximum_1/y*
_output_shapes
: *
T0*9
_class/
-+loc:@batch_normalization_1/moments/variance
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
Truncate( *
_output_shapes
: *

DstT0
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
Ptraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/subSubdense_1/Relu*batch_normalization_1/moments/StopGradientL^training/Adam/gradients/batch_normalization_1/moments/variance_grad/truediv*(
_output_shapes
:����������*
T0*B
_class8
64loc:@batch_normalization_1/moments/SquaredDifference
�
Rtraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/mul_1MulPtraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/MulPtraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/sub*
T0*B
_class8
64loc:@batch_normalization_1/moments/SquaredDifference*(
_output_shapes
:����������
�
Rtraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/ShapeShapedense_1/Relu*
_output_shapes
:*
T0*
out_type0*B
_class8
64loc:@batch_normalization_1/moments/SquaredDifference
�
Ttraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/Shape_1Shape*batch_normalization_1/moments/StopGradient*
T0*
out_type0*B
_class8
64loc:@batch_normalization_1/moments/SquaredDifference*
_output_shapes
:
�
btraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgsRtraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/ShapeTtraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/Shape_1*
T0*B
_class8
64loc:@batch_normalization_1/moments/SquaredDifference*2
_output_shapes 
:���������:���������
�
Ptraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/SumSumRtraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/mul_1btraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*B
_class8
64loc:@batch_normalization_1/moments/SquaredDifference
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
Gtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Shape_1Const*
valueB:*5
_class+
)'loc:@batch_normalization_1/moments/mean*
dtype0*
_output_shapes
:
�
Ktraining/Adam/gradients/batch_normalization_1/moments/mean_grad/range/startConst*
value	B : *5
_class+
)'loc:@batch_normalization_1/moments/mean*
dtype0*
_output_shapes
: 
�
Ktraining/Adam/gradients/batch_normalization_1/moments/mean_grad/range/deltaConst*
value	B :*5
_class+
)'loc:@batch_normalization_1/moments/mean*
dtype0*
_output_shapes
: 
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
Dtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/FillFillGtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Shape_1Jtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Fill/value*
T0*

index_type0*5
_class+
)'loc:@batch_normalization_1/moments/mean*
_output_shapes
:
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
Gtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/MaximumMaximumMtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/DynamicStitchItraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Maximum/y*
_output_shapes
:*
T0*5
_class+
)'loc:@batch_normalization_1/moments/mean
�
Htraining/Adam/gradients/batch_normalization_1/moments/mean_grad/floordivFloorDivEtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/ShapeGtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Maximum*
_output_shapes
:*
T0*5
_class+
)'loc:@batch_normalization_1/moments/mean
�
Gtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/ReshapeReshapeJtraining/Adam/gradients/batch_normalization_1/moments/Squeeze_grad/ReshapeMtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/DynamicStitch*
T0*
Tshape0*5
_class+
)'loc:@batch_normalization_1/moments/mean*0
_output_shapes
:������������������
�
Dtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/TileTileGtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/ReshapeHtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/floordiv*
T0*5
_class+
)'loc:@batch_normalization_1/moments/mean*0
_output_shapes
:������������������*

Tmultiples0
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
dtype0*
_output_shapes
:*
valueB"   �   *5
_class+
)'loc:@batch_normalization_1/moments/mean
�
Etraining/Adam/gradients/batch_normalization_1/moments/mean_grad/ConstConst*
valueB: *5
_class+
)'loc:@batch_normalization_1/moments/mean*
dtype0*
_output_shapes
:
�
Dtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/ProdProdGtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Shape_2Etraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Const*
T0*5
_class+
)'loc:@batch_normalization_1/moments/mean*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
Gtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Const_1Const*
valueB: *5
_class+
)'loc:@batch_normalization_1/moments/mean*
dtype0*
_output_shapes
:
�
Ftraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Prod_1ProdGtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Shape_3Gtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Const_1*
T0*5
_class+
)'loc:@batch_normalization_1/moments/mean*
_output_shapes
: *

Tidx0*
	keep_dims( 
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
Jtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/floordiv_1FloorDivDtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/ProdItraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Maximum_1*
_output_shapes
: *
T0*5
_class+
)'loc:@batch_normalization_1/moments/mean
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
2training/Adam/gradients/dense_1/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_1/Relu_grad/ReluGraddense_1/MatMul/ReadVariableOp*
T0*!
_class
loc:@dense_1/MatMul*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
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
training/Adam/addAddV2training/Adam/Casttraining/Adam/add/y*
_output_shapes
: *
T0
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
training/Adam/sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
a
training/Adam/subSubtraining/Adam/sub/xtraining/Adam/Pow*
T0*
_output_shapes
: 
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
training/Adam/Pow_1Pow"training/Adam/Pow_1/ReadVariableOptraining/Adam/add*
_output_shapes
: *
T0
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
valueB"   �   *
dtype0*
_output_shapes
:
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
training/Adam/m_2_1VarHandleOp*&
_class
loc:@training/Adam/m_2_1*
	container *
shape:�*
dtype0*
_output_shapes
: *$
shared_nametraining/Adam/m_2_1
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
training/Adam/m_4_1VarHandleOp*$
shared_nametraining/Adam/m_4_1*&
_class
loc:@training/Adam/m_4_1*
	container *
shape:
��*
dtype0*
_output_shapes
: 
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
training/Adam/m_5_1VarHandleOp*&
_class
loc:@training/Adam/m_5_1*
	container *
shape:�*
dtype0*
_output_shapes
: *$
shared_nametraining/Adam/m_5_1
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
training/Adam/m_7Const*
valueB�*    *
dtype0*
_output_shapes	
:�
�
training/Adam/m_7_1VarHandleOp*
	container *
shape:�*
dtype0*
_output_shapes
: *$
shared_nametraining/Adam/m_7_1*&
_class
loc:@training/Adam/m_7_1
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
training/Adam/m_8/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/m_8Fill!training/Adam/m_8/shape_as_tensortraining/Adam/m_8/Const* 
_output_shapes
:
��*
T0*

index_type0
�
training/Adam/m_8_1VarHandleOp*&
_class
loc:@training/Adam/m_8_1*
	container *
shape:
��*
dtype0*
_output_shapes
: *$
shared_nametraining/Adam/m_8_1
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
training/Adam/m_9_1VarHandleOp*&
_class
loc:@training/Adam/m_9_1*
	container *
shape:�*
dtype0*
_output_shapes
: *$
shared_nametraining/Adam/m_9_1
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
training/Adam/m_10Const*
dtype0*
_output_shapes	
:�*
valueB�*    
�
training/Adam/m_10_1VarHandleOp*
dtype0*
_output_shapes
: *%
shared_nametraining/Adam/m_10_1*'
_class
loc:@training/Adam/m_10_1*
	container *
shape:�
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
training/Adam/m_12Fill"training/Adam/m_12/shape_as_tensortraining/Adam/m_12/Const*
T0*

index_type0* 
_output_shapes
:
��
�
training/Adam/m_12_1VarHandleOp*'
_class
loc:@training/Adam/m_12_1*
	container *
shape:
��*
dtype0*
_output_shapes
: *%
shared_nametraining/Adam/m_12_1
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
training/Adam/m_13_1VarHandleOp*
dtype0*
_output_shapes
: *%
shared_nametraining/Adam/m_13_1*'
_class
loc:@training/Adam/m_13_1*
	container *
shape:�
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
training/Adam/m_14_1VarHandleOp*%
shared_nametraining/Adam/m_14_1*'
_class
loc:@training/Adam/m_14_1*
	container *
shape:�*
dtype0*
_output_shapes
: 
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
training/Adam/m_16/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
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
training/Adam/v_1_1VarHandleOp*
dtype0*
_output_shapes
: *$
shared_nametraining/Adam/v_1_1*&
_class
loc:@training/Adam/v_1_1*
	container *
shape:�
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
training/Adam/v_3Const*
dtype0*
_output_shapes	
:�*
valueB�*    
�
training/Adam/v_3_1VarHandleOp*
dtype0*
_output_shapes
: *$
shared_nametraining/Adam/v_3_1*&
_class
loc:@training/Adam/v_3_1*
	container *
shape:�
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
training/Adam/v_4/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
�
training/Adam/v_4Fill!training/Adam/v_4/shape_as_tensortraining/Adam/v_4/Const* 
_output_shapes
:
��*
T0*

index_type0
�
training/Adam/v_4_1VarHandleOp*&
_class
loc:@training/Adam/v_4_1*
	container *
shape:
��*
dtype0*
_output_shapes
: *$
shared_nametraining/Adam/v_4_1
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
training/Adam/v_5_1VarHandleOp*
dtype0*
_output_shapes
: *$
shared_nametraining/Adam/v_5_1*&
_class
loc:@training/Adam/v_5_1*
	container *
shape:�
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
training/Adam/v_6_1VarHandleOp*
	container *
shape:�*
dtype0*
_output_shapes
: *$
shared_nametraining/Adam/v_6_1*&
_class
loc:@training/Adam/v_6_1
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
training/Adam/v_7_1VarHandleOp*
dtype0*
_output_shapes
: *$
shared_nametraining/Adam/v_7_1*&
_class
loc:@training/Adam/v_7_1*
	container *
shape:�
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
training/Adam/v_8_1VarHandleOp*$
shared_nametraining/Adam/v_8_1*&
_class
loc:@training/Adam/v_8_1*
	container *
shape:
��*
dtype0*
_output_shapes
: 
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
training/Adam/v_10_1VarHandleOp*'
_class
loc:@training/Adam/v_10_1*
	container *
shape:�*
dtype0*
_output_shapes
: *%
shared_nametraining/Adam/v_10_1
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
training/Adam/v_12/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/v_12Fill"training/Adam/v_12/shape_as_tensortraining/Adam/v_12/Const*
T0*

index_type0* 
_output_shapes
:
��
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
training/Adam/v_13_1VarHandleOp*'
_class
loc:@training/Adam/v_13_1*
	container *
shape:�*
dtype0*
_output_shapes
: *%
shared_nametraining/Adam/v_13_1
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
training/Adam/v_15_1VarHandleOp*%
shared_nametraining/Adam/v_15_1*'
_class
loc:@training/Adam/v_15_1*
	container *
shape:�*
dtype0*
_output_shapes
: 
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
training/Adam/v_16/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/v_16Fill"training/Adam/v_16/shape_as_tensortraining/Adam/v_16/Const*
T0*

index_type0*
_output_shapes
:	�+
�
training/Adam/v_16_1VarHandleOp*
	container *
shape:	�+*
dtype0*
_output_shapes
: *%
shared_nametraining/Adam/v_16_1*'
_class
loc:@training/Adam/v_16_1
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
training/Adam/vhat_0Fill$training/Adam/vhat_0/shape_as_tensortraining/Adam/vhat_0/Const*
_output_shapes
:*
T0*

index_type0
�
training/Adam/vhat_0_1VarHandleOp*
dtype0*
_output_shapes
: *'
shared_nametraining/Adam/vhat_0_1*)
_class
loc:@training/Adam/vhat_0_1*
	container *
shape:
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
training/Adam/vhat_1/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
�
training/Adam/vhat_1Fill$training/Adam/vhat_1/shape_as_tensortraining/Adam/vhat_1/Const*
_output_shapes
:*
T0*

index_type0
�
training/Adam/vhat_1_1VarHandleOp*
shape:*
dtype0*
_output_shapes
: *'
shared_nametraining/Adam/vhat_1_1*)
_class
loc:@training/Adam/vhat_1_1*
	container 
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
training/Adam/vhat_2/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
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
training/Adam/vhat_3Fill$training/Adam/vhat_3/shape_as_tensortraining/Adam/vhat_3/Const*
T0*

index_type0*
_output_shapes
:
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
training/Adam/vhat_4/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
�
training/Adam/vhat_4Fill$training/Adam/vhat_4/shape_as_tensortraining/Adam/vhat_4/Const*
_output_shapes
:*
T0*

index_type0
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
training/Adam/vhat_5/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/vhat_5Fill$training/Adam/vhat_5/shape_as_tensortraining/Adam/vhat_5/Const*
_output_shapes
:*
T0*

index_type0
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
training/Adam/vhat_6_1VarHandleOp*
shape:*
dtype0*
_output_shapes
: *'
shared_nametraining/Adam/vhat_6_1*)
_class
loc:@training/Adam/vhat_6_1*
	container 
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
training/Adam/vhat_7_1VarHandleOp*'
shared_nametraining/Adam/vhat_7_1*)
_class
loc:@training/Adam/vhat_7_1*
	container *
shape:*
dtype0*
_output_shapes
: 
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
	container *
shape:*
dtype0*
_output_shapes
: *'
shared_nametraining/Adam/vhat_8_1*)
_class
loc:@training/Adam/vhat_8_1
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
$training/Adam/vhat_9/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
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
training/Adam/vhat_10Fill%training/Adam/vhat_10/shape_as_tensortraining/Adam/vhat_10/Const*
T0*

index_type0*
_output_shapes
:
�
training/Adam/vhat_10_1VarHandleOp*
dtype0*
_output_shapes
: *(
shared_nametraining/Adam/vhat_10_1**
_class 
loc:@training/Adam/vhat_10_1*
	container *
shape:
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
training/Adam/vhat_11_1VarHandleOp*
dtype0*
_output_shapes
: *(
shared_nametraining/Adam/vhat_11_1**
_class 
loc:@training/Adam/vhat_11_1*
	container *
shape:
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
%training/Adam/vhat_13/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
`
training/Adam/vhat_13/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
�
training/Adam/vhat_13Fill%training/Adam/vhat_13/shape_as_tensortraining/Adam/vhat_13/Const*
_output_shapes
:*
T0*

index_type0
�
training/Adam/vhat_13_1VarHandleOp*(
shared_nametraining/Adam/vhat_13_1**
_class 
loc:@training/Adam/vhat_13_1*
	container *
shape:*
dtype0*
_output_shapes
: 
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
%training/Adam/vhat_14/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
`
training/Adam/vhat_14/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
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
training/Adam/vhat_15_1VarHandleOp*
dtype0*
_output_shapes
: *(
shared_nametraining/Adam/vhat_15_1**
_class 
loc:@training/Adam/vhat_15_1*
	container *
shape:
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
training/Adam/vhat_17/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/vhat_17Fill%training/Adam/vhat_17/shape_as_tensortraining/Adam/vhat_17/Const*
T0*

index_type0*
_output_shapes
:
�
training/Adam/vhat_17_1VarHandleOp*
dtype0*
_output_shapes
: *(
shared_nametraining/Adam/vhat_17_1**
_class 
loc:@training/Adam/vhat_17_1*
	container *
shape:
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
training/Adam/sub_3/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
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
training/Adam/mul_4Multraining/Adam/sub_3training/Adam/Square*
T0* 
_output_shapes
:
��
q
training/Adam/add_2AddV2training/Adam/mul_3training/Adam/mul_4*
T0* 
_output_shapes
:
��
m
training/Adam/mul_5Multraining/Adam/multraining/Adam/add_1*
T0* 
_output_shapes
:
��
Z
training/Adam/Const_3Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_4Const*
dtype0*
_output_shapes
: *
valueB
 *  �
�
%training/Adam/clip_by_value_1/MinimumMinimumtraining/Adam/add_2training/Adam/Const_4*
T0* 
_output_shapes
:
��
�
training/Adam/clip_by_value_1Maximum%training/Adam/clip_by_value_1/Minimumtraining/Adam/Const_3* 
_output_shapes
:
��*
T0
f
training/Adam/Sqrt_1Sqrttraining/Adam/clip_by_value_1* 
_output_shapes
:
��*
T0
Z
training/Adam/add_3/yConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
t
training/Adam/add_3AddV2training/Adam/Sqrt_1training/Adam/add_3/y* 
_output_shapes
:
��*
T0
w
training/Adam/truediv_1RealDivtraining/Adam/mul_5training/Adam/add_3*
T0* 
_output_shapes
:
��
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
training/Adam/mul_6Multraining/Adam/ReadVariableOp_10"training/Adam/mul_6/ReadVariableOp*
_output_shapes	
:�*
T0
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
training/Adam/mul_7Multraining/Adam/sub_58training/Adam/gradients/dense_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:�
l
training/Adam/add_4AddV2training/Adam/mul_6training/Adam/mul_7*
T0*
_output_shapes	
:�
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
training/Adam/mul_9Multraining/Adam/sub_6training/Adam/Square_1*
T0*
_output_shapes	
:�
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
training/Adam/sub_8/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
s
training/Adam/sub_8Subtraining/Adam/sub_8/xtraining/Adam/ReadVariableOp_19*
_output_shapes
: *
T0
w
training/Adam/mul_12Multraining/Adam/sub_8training/Adam/gradients/AddN_20*
_output_shapes	
:�*
T0
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
training/Adam/sub_11/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
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
training/Adam/add_11AddV2training/Adam/mul_18training/Adam/mul_19*
T0*
_output_shapes	
:�
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
training/Adam/sub_15/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
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
%training/Adam/clip_by_value_5/MinimumMinimumtraining/Adam/add_14training/Adam/Const_12* 
_output_shapes
:
��*
T0
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
training/Adam/add_15/yConst*
dtype0*
_output_shapes
: *
valueB
 *���3
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
training/Adam/mul_27Multraining/Adam/sub_178training/Adam/gradients/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�*
T0
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
training/Adam/mul_31Multraining/Adam/ReadVariableOp_50#training/Adam/mul_31/ReadVariableOp*
_output_shapes	
:�*
T0
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
training/Adam/sub_21/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
u
training/Adam/sub_21Subtraining/Adam/sub_21/xtraining/Adam/ReadVariableOp_53*
T0*
_output_shapes
: 
g
training/Adam/Square_6Squaretraining/Adam/gradients/AddN_15*
T0*
_output_shapes	
:�
o
training/Adam/mul_34Multraining/Adam/sub_21training/Adam/Square_6*
_output_shapes	
:�*
T0
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
training/Adam/Const_16Const*
dtype0*
_output_shapes
: *
valueB
 *  �
�
%training/Adam/clip_by_value_7/MinimumMinimumtraining/Adam/add_20training/Adam/Const_16*
_output_shapes	
:�*
T0
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
training/Adam/add_21AddV2training/Adam/Sqrt_7training/Adam/add_21/y*
_output_shapes	
:�*
T0
t
training/Adam/truediv_7RealDivtraining/Adam/mul_35training/Adam/add_21*
_output_shapes	
:�*
T0
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
training/Adam/add_23AddV2training/Adam/mul_38training/Adam/mul_39*
T0*
_output_shapes	
:�
j
training/Adam/mul_40Multraining/Adam/multraining/Adam/add_22*
T0*
_output_shapes	
:�
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
training/Adam/add_24/yConst*
dtype0*
_output_shapes
: *
valueB
 *���3
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
training/Adam/sub_26/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
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
training/Adam/Const_19Const*
dtype0*
_output_shapes
: *
valueB
 *    
[
training/Adam/Const_20Const*
valueB
 *  �*
dtype0*
_output_shapes
: 
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
training/Adam/add_27/yConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
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
training/Adam/sub_28Subtraining/Adam/ReadVariableOp_70training/Adam/truediv_9* 
_output_shapes
:
��*
T0
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
training/Adam/sub_29/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
u
training/Adam/sub_29Subtraining/Adam/sub_29/xtraining/Adam/ReadVariableOp_75*
_output_shapes
: *
T0
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
training/Adam/mul_48Multraining/Adam/ReadVariableOp_76#training/Adam/mul_48/ReadVariableOp*
_output_shapes	
:�*
T0
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
training/Adam/Const_21Const*
dtype0*
_output_shapes
: *
valueB
 *    
[
training/Adam/Const_22Const*
dtype0*
_output_shapes
: *
valueB
 *  �
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
training/Adam/Sqrt_10Sqrttraining/Adam/clip_by_value_10*
_output_shapes	
:�*
T0
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
training/Adam/add_31AddV2training/Adam/mul_51training/Adam/mul_52*
T0*
_output_shapes	
:�
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
training/Adam/Square_10Squaretraining/Adam/gradients/AddN_9*
_output_shapes	
:�*
T0
p
training/Adam/mul_54Multraining/Adam/sub_33training/Adam/Square_10*
_output_shapes	
:�*
T0
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
training/Adam/Const_24Const*
valueB
 *  �*
dtype0*
_output_shapes
: 
�
&training/Adam/clip_by_value_11/MinimumMinimumtraining/Adam/add_32training/Adam/Const_24*
T0*
_output_shapes	
:�
�
training/Adam/clip_by_value_11Maximum&training/Adam/clip_by_value_11/Minimumtraining/Adam/Const_23*
T0*
_output_shapes	
:�
c
training/Adam/Sqrt_11Sqrttraining/Adam/clip_by_value_11*
_output_shapes	
:�*
T0
[
training/Adam/add_33/yConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
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
training/Adam/mul_56Multraining/Adam/ReadVariableOp_90#training/Adam/mul_56/ReadVariableOp*
_output_shapes	
:�*
T0
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
training/Adam/truediv_12RealDivtraining/Adam/mul_60training/Adam/add_36*
_output_shapes	
:�*
T0
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
training/Adam/mul_61Multraining/Adam/ReadVariableOp_98#training/Adam/mul_61/ReadVariableOp*
T0* 
_output_shapes
:
��
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
training/Adam/sub_38Subtraining/Adam/sub_38/xtraining/Adam/ReadVariableOp_99*
_output_shapes
: *
T0
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
training/Adam/sub_39/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
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
training/Adam/add_38AddV2training/Adam/mul_63training/Adam/mul_64*
T0* 
_output_shapes
:
��
o
training/Adam/mul_65Multraining/Adam/multraining/Adam/add_37*
T0* 
_output_shapes
:
��
[
training/Adam/Const_27Const*
dtype0*
_output_shapes
: *
valueB
 *    
[
training/Adam/Const_28Const*
valueB
 *  �*
dtype0*
_output_shapes
: 
�
&training/Adam/clip_by_value_13/MinimumMinimumtraining/Adam/add_38training/Adam/Const_28* 
_output_shapes
:
��*
T0
�
training/Adam/clip_by_value_13Maximum&training/Adam/clip_by_value_13/Minimumtraining/Adam/Const_27*
T0* 
_output_shapes
:
��
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
training/Adam/truediv_13RealDivtraining/Adam/mul_65training/Adam/add_39* 
_output_shapes
:
��*
T0
q
 training/Adam/ReadVariableOp_102ReadVariableOpdense_4/kernel*
dtype0* 
_output_shapes
:
��
�
training/Adam/sub_40Sub training/Adam/ReadVariableOp_102training/Adam/truediv_13* 
_output_shapes
:
��*
T0
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
training/Adam/sub_41Subtraining/Adam/sub_41/x training/Adam/ReadVariableOp_107*
_output_shapes
: *
T0
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
training/Adam/mul_69Multraining/Adam/sub_42training/Adam/Square_13*
T0*
_output_shapes	
:�
o
training/Adam/add_41AddV2training/Adam/mul_68training/Adam/mul_69*
T0*
_output_shapes	
:�
j
training/Adam/mul_70Multraining/Adam/multraining/Adam/add_40*
_output_shapes	
:�*
T0
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
training/Adam/add_42/yConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
r
training/Adam/add_42AddV2training/Adam/Sqrt_14training/Adam/add_42/y*
_output_shapes	
:�*
T0
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
training/Adam/mul_72Multraining/Adam/sub_44training/Adam/gradients/AddN_3*
_output_shapes	
:�*
T0
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
training/Adam/mul_73Mul training/Adam/ReadVariableOp_116#training/Adam/mul_73/ReadVariableOp*
T0*
_output_shapes	
:�
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
training/Adam/add_44AddV2training/Adam/mul_73training/Adam/mul_74*
_output_shapes	
:�*
T0
j
training/Adam/mul_75Multraining/Adam/multraining/Adam/add_43*
_output_shapes	
:�*
T0
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
&training/Adam/clip_by_value_15/MinimumMinimumtraining/Adam/add_44training/Adam/Const_32*
_output_shapes	
:�*
T0
�
training/Adam/clip_by_value_15Maximum&training/Adam/clip_by_value_15/Minimumtraining/Adam/Const_31*
T0*
_output_shapes	
:�
c
training/Adam/Sqrt_15Sqrttraining/Adam/clip_by_value_15*
_output_shapes	
:�*
T0
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
training/Adam/mul_76Mul training/Adam/ReadVariableOp_122#training/Adam/mul_76/ReadVariableOp*
_output_shapes	
:�*
T0
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
training/Adam/clip_by_value_16Maximum&training/Adam/clip_by_value_16/Minimumtraining/Adam/Const_33*
_output_shapes	
:�*
T0
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
training/Adam/sub_49Sub training/Adam/ReadVariableOp_126training/Adam/truediv_16*
T0*
_output_shapes	
:�
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
training/Adam/clip_by_value_17Maximum&training/Adam/clip_by_value_17/Minimumtraining/Adam/Const_35*
_output_shapes
:	�+*
T0
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
training/Adam/sub_53Subtraining/Adam/sub_53/x training/Adam/ReadVariableOp_139*
T0*
_output_shapes
: 
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
training/Adam/Square_17Square8training/Adam/gradients/dense_5/BiasAdd_grad/BiasAddGrad*
_output_shapes
:+*
T0
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
training/Adam/clip_by_value_18Maximum&training/Adam/clip_by_value_18/Minimumtraining/Adam/Const_37*
_output_shapes
:+*
T0
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
training/Adam/add_54AddV2training/Adam/Sqrt_18training/Adam/add_54/y*
_output_shapes
:+*
T0
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
_
training/VarIsInitializedOpVarIsInitializedOptraining/Adam/m_12_1*
_output_shapes
: 
`
training/VarIsInitializedOp_1VarIsInitializedOptraining/Adam/v_2_1*
_output_shapes
: 
c
training/VarIsInitializedOp_2VarIsInitializedOptraining/Adam/vhat_3_1*
_output_shapes
: 
c
training/VarIsInitializedOp_3VarIsInitializedOptraining/Adam/vhat_9_1*
_output_shapes
: 
d
training/VarIsInitializedOp_4VarIsInitializedOptraining/Adam/vhat_15_1*
_output_shapes
: 
[
training/VarIsInitializedOp_5VarIsInitializedOpdense_2/kernel*
_output_shapes
: 
R
training/VarIsInitializedOp_6VarIsInitializedOptotal*
_output_shapes
: 
`
training/VarIsInitializedOp_7VarIsInitializedOptraining/Adam/m_0_1*
_output_shapes
: 
a
training/VarIsInitializedOp_8VarIsInitializedOptraining/Adam/v_17_1*
_output_shapes
: 
c
training/VarIsInitializedOp_9VarIsInitializedOptraining/Adam/vhat_6_1*
_output_shapes
: 
e
training/VarIsInitializedOp_10VarIsInitializedOptraining/Adam/vhat_12_1*
_output_shapes
: 
]
training/VarIsInitializedOp_11VarIsInitializedOpAdam/iterations*
_output_shapes
: 
a
training/VarIsInitializedOp_12VarIsInitializedOptraining/Adam/m_2_1*
_output_shapes
: 
b
training/VarIsInitializedOp_13VarIsInitializedOptraining/Adam/v_11_1*
_output_shapes
: 
d
training/VarIsInitializedOp_14VarIsInitializedOptraining/Adam/vhat_0_1*
_output_shapes
: 
e
training/VarIsInitializedOp_15VarIsInitializedOptraining/Adam/vhat_14_1*
_output_shapes
: 
a
training/VarIsInitializedOp_16VarIsInitializedOptraining/Adam/v_7_1*
_output_shapes
: 
Z
training/VarIsInitializedOp_17VarIsInitializedOpdense_1/bias*
_output_shapes
: 
o
training/VarIsInitializedOp_18VarIsInitializedOp!batch_normalization_1/moving_mean*
_output_shapes
: 
s
training/VarIsInitializedOp_19VarIsInitializedOp%batch_normalization_1/moving_variance*
_output_shapes
: 
s
training/VarIsInitializedOp_20VarIsInitializedOp%batch_normalization_3/moving_variance*
_output_shapes
: 
h
training/VarIsInitializedOp_21VarIsInitializedOpbatch_normalization_4/beta*
_output_shapes
: 
S
training/VarIsInitializedOp_22VarIsInitializedOpcount*
_output_shapes
: 
a
training/VarIsInitializedOp_23VarIsInitializedOptraining/Adam/m_1_1*
_output_shapes
: 
a
training/VarIsInitializedOp_24VarIsInitializedOptraining/Adam/m_7_1*
_output_shapes
: 
b
training/VarIsInitializedOp_25VarIsInitializedOptraining/Adam/m_11_1*
_output_shapes
: 
a
training/VarIsInitializedOp_26VarIsInitializedOptraining/Adam/v_1_1*
_output_shapes
: 
a
training/VarIsInitializedOp_27VarIsInitializedOptraining/Adam/v_4_1*
_output_shapes
: 
Z
training/VarIsInitializedOp_28VarIsInitializedOpdense_2/bias*
_output_shapes
: 
\
training/VarIsInitializedOp_29VarIsInitializedOpdense_3/kernel*
_output_shapes
: 
\
training/VarIsInitializedOp_30VarIsInitializedOpdense_5/kernel*
_output_shapes
: 
d
training/VarIsInitializedOp_31VarIsInitializedOptraining/Adam/vhat_4_1*
_output_shapes
: 
e
training/VarIsInitializedOp_32VarIsInitializedOptraining/Adam/vhat_10_1*
_output_shapes
: 
e
training/VarIsInitializedOp_33VarIsInitializedOptraining/Adam/vhat_16_1*
_output_shapes
: 
d
training/VarIsInitializedOp_34VarIsInitializedOptraining/Adam/vhat_1_1*
_output_shapes
: 
d
training/VarIsInitializedOp_35VarIsInitializedOptraining/Adam/vhat_7_1*
_output_shapes
: 
i
training/VarIsInitializedOp_36VarIsInitializedOpbatch_normalization_1/gamma*
_output_shapes
: 
a
training/VarIsInitializedOp_37VarIsInitializedOptraining/Adam/m_8_1*
_output_shapes
: 
e
training/VarIsInitializedOp_38VarIsInitializedOptraining/Adam/vhat_13_1*
_output_shapes
: 
a
training/VarIsInitializedOp_39VarIsInitializedOptraining/Adam/m_3_1*
_output_shapes
: 
Y
training/VarIsInitializedOp_40VarIsInitializedOpAdam/beta_2*
_output_shapes
: 
a
training/VarIsInitializedOp_41VarIsInitializedOptraining/Adam/v_6_1*
_output_shapes
: 
o
training/VarIsInitializedOp_42VarIsInitializedOp!batch_normalization_3/moving_mean*
_output_shapes
: 
i
training/VarIsInitializedOp_43VarIsInitializedOpbatch_normalization_4/gamma*
_output_shapes
: 
Z
training/VarIsInitializedOp_44VarIsInitializedOpdense_5/bias*
_output_shapes
: 
b
training/VarIsInitializedOp_45VarIsInitializedOptraining/Adam/m_15_1*
_output_shapes
: 
a
training/VarIsInitializedOp_46VarIsInitializedOptraining/Adam/v_5_1*
_output_shapes
: 
Z
training/VarIsInitializedOp_47VarIsInitializedOpdense_4/bias*
_output_shapes
: 
b
training/VarIsInitializedOp_48VarIsInitializedOptraining/Adam/v_12_1*
_output_shapes
: 
i
training/VarIsInitializedOp_49VarIsInitializedOpbatch_normalization_3/gamma*
_output_shapes
: 
a
training/VarIsInitializedOp_50VarIsInitializedOptraining/Adam/m_4_1*
_output_shapes
: 
b
training/VarIsInitializedOp_51VarIsInitializedOptraining/Adam/m_14_1*
_output_shapes
: 
b
training/VarIsInitializedOp_52VarIsInitializedOptraining/Adam/m_16_1*
_output_shapes
: 
d
training/VarIsInitializedOp_53VarIsInitializedOptraining/Adam/vhat_5_1*
_output_shapes
: 
i
training/VarIsInitializedOp_54VarIsInitializedOpbatch_normalization_2/gamma*
_output_shapes
: 
b
training/VarIsInitializedOp_55VarIsInitializedOptraining/Adam/v_15_1*
_output_shapes
: 
e
training/VarIsInitializedOp_56VarIsInitializedOptraining/Adam/vhat_11_1*
_output_shapes
: 
e
training/VarIsInitializedOp_57VarIsInitializedOptraining/Adam/vhat_17_1*
_output_shapes
: 
d
training/VarIsInitializedOp_58VarIsInitializedOptraining/Adam/vhat_8_1*
_output_shapes
: 
d
training/VarIsInitializedOp_59VarIsInitializedOptraining/Adam/vhat_2_1*
_output_shapes
: 
b
training/VarIsInitializedOp_60VarIsInitializedOptraining/Adam/m_10_1*
_output_shapes
: 
a
training/VarIsInitializedOp_61VarIsInitializedOptraining/Adam/v_8_1*
_output_shapes
: 
b
training/VarIsInitializedOp_62VarIsInitializedOptraining/Adam/v_16_1*
_output_shapes
: 
b
training/VarIsInitializedOp_63VarIsInitializedOptraining/Adam/v_14_1*
_output_shapes
: 
Y
training/VarIsInitializedOp_64VarIsInitializedOpAdam/beta_1*
_output_shapes
: 
a
training/VarIsInitializedOp_65VarIsInitializedOptraining/Adam/m_6_1*
_output_shapes
: 
\
training/VarIsInitializedOp_66VarIsInitializedOpdense_4/kernel*
_output_shapes
: 
h
training/VarIsInitializedOp_67VarIsInitializedOpbatch_normalization_1/beta*
_output_shapes
: 
h
training/VarIsInitializedOp_68VarIsInitializedOpbatch_normalization_2/beta*
_output_shapes
: 
`
training/VarIsInitializedOp_69VarIsInitializedOpAdam/learning_rate*
_output_shapes
: 
X
training/VarIsInitializedOp_70VarIsInitializedOp
Adam/decay*
_output_shapes
: 
a
training/VarIsInitializedOp_71VarIsInitializedOptraining/Adam/m_5_1*
_output_shapes
: 
a
training/VarIsInitializedOp_72VarIsInitializedOptraining/Adam/m_9_1*
_output_shapes
: 
s
training/VarIsInitializedOp_73VarIsInitializedOp%batch_normalization_4/moving_variance*
_output_shapes
: 
a
training/VarIsInitializedOp_74VarIsInitializedOptraining/Adam/v_3_1*
_output_shapes
: 
b
training/VarIsInitializedOp_75VarIsInitializedOptraining/Adam/m_13_1*
_output_shapes
: 
b
training/VarIsInitializedOp_76VarIsInitializedOptraining/Adam/v_10_1*
_output_shapes
: 
a
training/VarIsInitializedOp_77VarIsInitializedOptraining/Adam/v_0_1*
_output_shapes
: 
a
training/VarIsInitializedOp_78VarIsInitializedOptraining/Adam/v_9_1*
_output_shapes
: 
Z
training/VarIsInitializedOp_79VarIsInitializedOpdense_3/bias*
_output_shapes
: 
h
training/VarIsInitializedOp_80VarIsInitializedOpbatch_normalization_3/beta*
_output_shapes
: 
s
training/VarIsInitializedOp_81VarIsInitializedOp%batch_normalization_2/moving_variance*
_output_shapes
: 
o
training/VarIsInitializedOp_82VarIsInitializedOp!batch_normalization_2/moving_mean*
_output_shapes
: 
o
training/VarIsInitializedOp_83VarIsInitializedOp!batch_normalization_4/moving_mean*
_output_shapes
: 
b
training/VarIsInitializedOp_84VarIsInitializedOptraining/Adam/v_13_1*
_output_shapes
: 
\
training/VarIsInitializedOp_85VarIsInitializedOpdense_1/kernel*
_output_shapes
: 
b
training/VarIsInitializedOp_86VarIsInitializedOptraining/Adam/m_17_1*
_output_shapes
: 
�
training/initNoOp^Adam/beta_1/Assign^Adam/beta_2/Assign^Adam/decay/Assign^Adam/iterations/Assign^Adam/learning_rate/Assign"^batch_normalization_1/beta/Assign#^batch_normalization_1/gamma/Assign)^batch_normalization_1/moving_mean/Assign-^batch_normalization_1/moving_variance/Assign"^batch_normalization_2/beta/Assign#^batch_normalization_2/gamma/Assign)^batch_normalization_2/moving_mean/Assign-^batch_normalization_2/moving_variance/Assign"^batch_normalization_3/beta/Assign#^batch_normalization_3/gamma/Assign)^batch_normalization_3/moving_mean/Assign-^batch_normalization_3/moving_variance/Assign"^batch_normalization_4/beta/Assign#^batch_normalization_4/gamma/Assign)^batch_normalization_4/moving_mean/Assign-^batch_normalization_4/moving_variance/Assign^count/Assign^dense_1/bias/Assign^dense_1/kernel/Assign^dense_2/bias/Assign^dense_2/kernel/Assign^dense_3/bias/Assign^dense_3/kernel/Assign^dense_4/bias/Assign^dense_4/kernel/Assign^dense_5/bias/Assign^dense_5/kernel/Assign^total/Assign^training/Adam/m_0_1/Assign^training/Adam/m_10_1/Assign^training/Adam/m_11_1/Assign^training/Adam/m_12_1/Assign^training/Adam/m_13_1/Assign^training/Adam/m_14_1/Assign^training/Adam/m_15_1/Assign^training/Adam/m_16_1/Assign^training/Adam/m_17_1/Assign^training/Adam/m_1_1/Assign^training/Adam/m_2_1/Assign^training/Adam/m_3_1/Assign^training/Adam/m_4_1/Assign^training/Adam/m_5_1/Assign^training/Adam/m_6_1/Assign^training/Adam/m_7_1/Assign^training/Adam/m_8_1/Assign^training/Adam/m_9_1/Assign^training/Adam/v_0_1/Assign^training/Adam/v_10_1/Assign^training/Adam/v_11_1/Assign^training/Adam/v_12_1/Assign^training/Adam/v_13_1/Assign^training/Adam/v_14_1/Assign^training/Adam/v_15_1/Assign^training/Adam/v_16_1/Assign^training/Adam/v_17_1/Assign^training/Adam/v_1_1/Assign^training/Adam/v_2_1/Assign^training/Adam/v_3_1/Assign^training/Adam/v_4_1/Assign^training/Adam/v_5_1/Assign^training/Adam/v_6_1/Assign^training/Adam/v_7_1/Assign^training/Adam/v_8_1/Assign^training/Adam/v_9_1/Assign^training/Adam/vhat_0_1/Assign^training/Adam/vhat_10_1/Assign^training/Adam/vhat_11_1/Assign^training/Adam/vhat_12_1/Assign^training/Adam/vhat_13_1/Assign^training/Adam/vhat_14_1/Assign^training/Adam/vhat_15_1/Assign^training/Adam/vhat_16_1/Assign^training/Adam/vhat_17_1/Assign^training/Adam/vhat_1_1/Assign^training/Adam/vhat_2_1/Assign^training/Adam/vhat_3_1/Assign^training/Adam/vhat_4_1/Assign^training/Adam/vhat_5_1/Assign^training/Adam/vhat_6_1/Assign^training/Adam/vhat_7_1/Assign^training/Adam/vhat_8_1/Assign^training/Adam/vhat_9_1/Assign
�
training/group_depsNoOp^Mean*^batch_normalization_1/AssignSubVariableOp,^batch_normalization_1/AssignSubVariableOp_1*^batch_normalization_2/AssignSubVariableOp,^batch_normalization_2/AssignSubVariableOp_1*^batch_normalization_3/AssignSubVariableOp,^batch_normalization_3/AssignSubVariableOp_1*^batch_normalization_4/AssignSubVariableOp,^batch_normalization_4/AssignSubVariableOp_1%^metrics/accuracy/AssignAddVariableOp'^metrics/accuracy/AssignAddVariableOp_1"^training/Adam/AssignAddVariableOp^training/Adam/AssignVariableOp!^training/Adam/AssignVariableOp_1"^training/Adam/AssignVariableOp_10"^training/Adam/AssignVariableOp_11"^training/Adam/AssignVariableOp_12"^training/Adam/AssignVariableOp_13"^training/Adam/AssignVariableOp_14"^training/Adam/AssignVariableOp_15"^training/Adam/AssignVariableOp_16"^training/Adam/AssignVariableOp_17"^training/Adam/AssignVariableOp_18"^training/Adam/AssignVariableOp_19!^training/Adam/AssignVariableOp_2"^training/Adam/AssignVariableOp_20"^training/Adam/AssignVariableOp_21"^training/Adam/AssignVariableOp_22"^training/Adam/AssignVariableOp_23"^training/Adam/AssignVariableOp_24"^training/Adam/AssignVariableOp_25"^training/Adam/AssignVariableOp_26"^training/Adam/AssignVariableOp_27"^training/Adam/AssignVariableOp_28"^training/Adam/AssignVariableOp_29!^training/Adam/AssignVariableOp_3"^training/Adam/AssignVariableOp_30"^training/Adam/AssignVariableOp_31"^training/Adam/AssignVariableOp_32"^training/Adam/AssignVariableOp_33"^training/Adam/AssignVariableOp_34"^training/Adam/AssignVariableOp_35"^training/Adam/AssignVariableOp_36"^training/Adam/AssignVariableOp_37"^training/Adam/AssignVariableOp_38"^training/Adam/AssignVariableOp_39!^training/Adam/AssignVariableOp_4"^training/Adam/AssignVariableOp_40"^training/Adam/AssignVariableOp_41"^training/Adam/AssignVariableOp_42"^training/Adam/AssignVariableOp_43"^training/Adam/AssignVariableOp_44"^training/Adam/AssignVariableOp_45"^training/Adam/AssignVariableOp_46"^training/Adam/AssignVariableOp_47"^training/Adam/AssignVariableOp_48"^training/Adam/AssignVariableOp_49!^training/Adam/AssignVariableOp_5"^training/Adam/AssignVariableOp_50"^training/Adam/AssignVariableOp_51"^training/Adam/AssignVariableOp_52"^training/Adam/AssignVariableOp_53!^training/Adam/AssignVariableOp_6!^training/Adam/AssignVariableOp_7!^training/Adam/AssignVariableOp_8!^training/Adam/AssignVariableOp_9
i

group_depsNoOp^Mean%^metrics/accuracy/AssignAddVariableOp'^metrics/accuracy/AssignAddVariableOp_1"� ���B     �(��	 c٠��AJ�
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
dtype0*(
_output_shapes
:����������*
shape:����������
m
dense_1/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"   �   
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
seed2���* 
_output_shapes
:
��*
seed���)*
T0
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
dense_1/random_uniformAdddense_1/random_uniform/muldense_1/random_uniform/min* 
_output_shapes
:
��*
T0
�
dense_1/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shared_namedense_1/kernel*!
_class
loc:@dense_1/kernel*
	container *
shape:
��
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
dense_1/ConstConst*
dtype0*
_output_shapes	
:�*
valueB�*    
�
dense_1/biasVarHandleOp*
	container *
shape:�*
dtype0*
_output_shapes
: *
shared_namedense_1/bias*
_class
loc:@dense_1/bias
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
batch_normalization_1/gammaVarHandleOp*,
shared_namebatch_normalization_1/gamma*.
_class$
" loc:@batch_normalization_1/gamma*
	container *
shape:�*
dtype0*
_output_shapes
: 
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
batch_normalization_1/Const_2Const*
dtype0*
_output_shapes	
:�*
valueB�*    
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
"batch_normalization_1/moments/meanMeandense_1/Relu4batch_normalization_1/moments/mean/reduction_indices*

Tidx0*
	keep_dims(*
T0*
_output_shapes
:	�
�
*batch_normalization_1/moments/StopGradientStopGradient"batch_normalization_1/moments/mean*
_output_shapes
:	�*
T0
�
/batch_normalization_1/moments/SquaredDifferenceSquaredDifferencedense_1/Relu*batch_normalization_1/moments/StopGradient*
T0*(
_output_shapes
:����������
�
8batch_normalization_1/moments/variance/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB: 
�
&batch_normalization_1/moments/varianceMean/batch_normalization_1/moments/SquaredDifference8batch_normalization_1/moments/variance/reduction_indices*
_output_shapes
:	�*

Tidx0*
	keep_dims(*
T0
�
%batch_normalization_1/moments/SqueezeSqueeze"batch_normalization_1/moments/mean*
_output_shapes	
:�*
squeeze_dims
 *
T0
�
'batch_normalization_1/moments/Squeeze_1Squeeze&batch_normalization_1/moments/variance*
squeeze_dims
 *
T0*
_output_shapes	
:�
j
%batch_normalization_1/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
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
%batch_normalization_1/batchnorm/mul_2Mul%batch_normalization_1/moments/Squeeze#batch_normalization_1/batchnorm/mul*
_output_shapes	
:�*
T0
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
#batch_normalization_1/strided_sliceStridedSlicebatch_normalization_1/Shape)batch_normalization_1/strided_slice/stack+batch_normalization_1/strided_slice/stack_1+batch_normalization_1/strided_slice/stack_2*
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
!batch_normalization_1/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
batch_normalization_1/rangeRange!batch_normalization_1/range/startbatch_normalization_1/Rank!batch_normalization_1/range/delta*

Tidx0*
_output_shapes
:
�
 batch_normalization_1/Prod/inputPack#batch_normalization_1/strided_slice*
T0*

axis *
N*
_output_shapes
:
�
batch_normalization_1/ProdProd batch_normalization_1/Prod/inputbatch_normalization_1/range*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
~
batch_normalization_1/CastCastbatch_normalization_1/Prod*

SrcT0*
Truncate( *

DstT0*
_output_shapes
: 
`
batch_normalization_1/sub/yConst*
dtype0*
_output_shapes
: *
valueB
 *� �?
z
batch_normalization_1/subSubbatch_normalization_1/Castbatch_normalization_1/sub/y*
_output_shapes
: *
T0
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
batch_normalization_1/sub_1Sub$batch_normalization_1/ReadVariableOp%batch_normalization_1/moments/Squeeze*
_output_shapes	
:�*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean
�
batch_normalization_1/mul_1Mulbatch_normalization_1/sub_1batch_normalization_1/Const_4*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes	
:�
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
batch_normalization_1/mul_2Mulbatch_normalization_1/sub_2batch_normalization_1/Const_5*
_output_shapes	
:�*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance
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
keras_learning_phase/inputConst*
value	B
 Z *
dtype0
*
_output_shapes
: 
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
"batch_normalization_1/cond/pred_idIdentitykeras_learning_phase*
T0
*
_output_shapes
: 
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
*batch_normalization_1/cond/batchnorm/mul_2Mul5batch_normalization_1/cond/batchnorm/ReadVariableOp_1(batch_normalization_1/cond/batchnorm/mul*
T0*
_output_shapes	
:�
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
seed2���* 
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
T0*
transpose_a( *(
_output_shapes
:����������
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
batch_normalization_2/ConstConst*
valueB�*  �?*
dtype0*
_output_shapes	
:�
�
batch_normalization_2/gammaVarHandleOp*
dtype0*
_output_shapes
: *,
shared_namebatch_normalization_2/gamma*.
_class$
" loc:@batch_normalization_2/gamma*
	container *
shape:�
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
batch_normalization_2/betaVarHandleOp*
dtype0*
_output_shapes
: *+
shared_namebatch_normalization_2/beta*-
_class#
!loc:@batch_normalization_2/beta*
	container *
shape:�
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
batch_normalization_2/Const_2Const*
dtype0*
_output_shapes	
:�*
valueB�*    
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
%batch_normalization_2/moments/SqueezeSqueeze"batch_normalization_2/moments/mean*
squeeze_dims
 *
T0*
_output_shapes	
:�
�
'batch_normalization_2/moments/Squeeze_1Squeeze&batch_normalization_2/moments/variance*
_output_shapes	
:�*
squeeze_dims
 *
T0
j
%batch_normalization_2/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *o�:
�
#batch_normalization_2/batchnorm/addAddV2'batch_normalization_2/moments/Squeeze_1%batch_normalization_2/batchnorm/add/y*
_output_shapes	
:�*
T0
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
%batch_normalization_2/batchnorm/mul_2Mul%batch_normalization_2/moments/Squeeze#batch_normalization_2/batchnorm/mul*
_output_shapes	
:�*
T0
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
)batch_normalization_2/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
u
+batch_normalization_2/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
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
batch_normalization_2/rangeRange!batch_normalization_2/range/startbatch_normalization_2/Rank!batch_normalization_2/range/delta*

Tidx0*
_output_shapes
:
�
 batch_normalization_2/Prod/inputPack#batch_normalization_2/strided_slice*
T0*

axis *
N*
_output_shapes
:
�
batch_normalization_2/ProdProd batch_normalization_2/Prod/inputbatch_normalization_2/range*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
~
batch_normalization_2/CastCastbatch_normalization_2/Prod*

SrcT0*
Truncate( *

DstT0*
_output_shapes
: 
`
batch_normalization_2/sub/yConst*
dtype0*
_output_shapes
: *
valueB
 *� �?
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
batch_normalization_2/mulMul'batch_normalization_2/moments/Squeeze_1batch_normalization_2/truediv*
_output_shapes	
:�*
T0
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
batch_normalization_2/sub_1Sub$batch_normalization_2/ReadVariableOp%batch_normalization_2/moments/Squeeze*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean*
_output_shapes	
:�
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
batch_normalization_2/sub_2Sub&batch_normalization_2/ReadVariableOp_2batch_normalization_2/mul*
_output_shapes	
:�*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance
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
&batch_normalization_2/ReadVariableOp_3ReadVariableOp%batch_normalization_2/moving_variance,^batch_normalization_2/AssignSubVariableOp_1*8
_class.
,*loc:@batch_normalization_2/moving_variance*
dtype0*
_output_shapes	
:�
z
!batch_normalization_2/cond/SwitchSwitchkeras_learning_phasekeras_learning_phase*
T0
*
_output_shapes
: : 
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
"batch_normalization_2/cond/pred_idIdentitykeras_learning_phase*
_output_shapes
: *
T0

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
dropout_1/cond/dropout/rateConst^dropout_1/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *   ?
�
dropout_1/cond/dropout/ShapeShape%dropout_1/cond/dropout/Shape/Switch:1*
T0*
out_type0*
_output_shapes
:
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
3dropout_1/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_1/cond/dropout/Shape*
T0*
dtype0*
seed2���*(
_output_shapes
:����������*
seed���)
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
dropout_1/cond/dropout/CastCast#dropout_1/cond/dropout/GreaterEqual*
Truncate( *

DstT0*(
_output_shapes
:����������*

SrcT0

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
N**
_output_shapes
:����������: *
T0
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
dtype0*
seed2���* 
_output_shapes
:
��
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
dense_3/kernelVarHandleOp*
shape:
��*
dtype0*
_output_shapes
: *
shared_namedense_3/kernel*!
_class
loc:@dense_3/kernel*
	container 
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
dense_3/MatMulMatMuldropout_1/cond/Mergedense_3/MatMul/ReadVariableOp*
transpose_a( *(
_output_shapes
:����������*
transpose_b( *
T0
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
batch_normalization_3/betaVarHandleOp*
dtype0*
_output_shapes
: *+
shared_namebatch_normalization_3/beta*-
_class#
!loc:@batch_normalization_3/beta*
	container *
shape:�
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
!batch_normalization_3/moving_meanVarHandleOp*
	container *
shape:�*
dtype0*
_output_shapes
: *2
shared_name#!batch_normalization_3/moving_mean*4
_class*
(&loc:@batch_normalization_3/moving_mean
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
%batch_normalization_3/moving_varianceVarHandleOp*8
_class.
,*loc:@batch_normalization_3/moving_variance*
	container *
shape:�*
dtype0*
_output_shapes
: *6
shared_name'%batch_normalization_3/moving_variance
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
:	�*

Tidx0*
	keep_dims(
�
*batch_normalization_3/moments/StopGradientStopGradient"batch_normalization_3/moments/mean*
_output_shapes
:	�*
T0
�
/batch_normalization_3/moments/SquaredDifferenceSquaredDifferencedense_3/Relu*batch_normalization_3/moments/StopGradient*
T0*(
_output_shapes
:����������
�
8batch_normalization_3/moments/variance/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB: 
�
&batch_normalization_3/moments/varianceMean/batch_normalization_3/moments/SquaredDifference8batch_normalization_3/moments/variance/reduction_indices*
T0*
_output_shapes
:	�*

Tidx0*
	keep_dims(
�
%batch_normalization_3/moments/SqueezeSqueeze"batch_normalization_3/moments/mean*
_output_shapes	
:�*
squeeze_dims
 *
T0
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
batch_normalization_3/ShapeShapedense_3/Relu*
_output_shapes
:*
T0*
out_type0
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
#batch_normalization_3/strided_sliceStridedSlicebatch_normalization_3/Shape)batch_normalization_3/strided_slice/stack+batch_normalization_3/strided_slice/stack_1+batch_normalization_3/strided_slice/stack_2*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask
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
!batch_normalization_3/range/startConst*
dtype0*
_output_shapes
: *
value	B : 
c
!batch_normalization_3/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
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
: *

Tidx0*
	keep_dims( *
T0
~
batch_normalization_3/CastCastbatch_normalization_3/Prod*
Truncate( *

DstT0*
_output_shapes
: *

SrcT0
`
batch_normalization_3/sub/yConst*
dtype0*
_output_shapes
: *
valueB
 *� �?
z
batch_normalization_3/subSubbatch_normalization_3/Castbatch_normalization_3/sub/y*
_output_shapes
: *
T0
�
batch_normalization_3/truedivRealDivbatch_normalization_3/Castbatch_normalization_3/sub*
_output_shapes
: *
T0
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
batch_normalization_3/sub_2Sub&batch_normalization_3/ReadVariableOp_2batch_normalization_3/mul*
_output_shapes	
:�*
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance
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
&batch_normalization_3/ReadVariableOp_3ReadVariableOp%batch_normalization_3/moving_variance,^batch_normalization_3/AssignSubVariableOp_1*8
_class.
,*loc:@batch_normalization_3/moving_variance*
dtype0*
_output_shapes	
:�
z
!batch_normalization_3/cond/SwitchSwitchkeras_learning_phasekeras_learning_phase*
T0
*
_output_shapes
: : 
u
#batch_normalization_3/cond/switch_tIdentity#batch_normalization_3/cond/Switch:1*
T0
*
_output_shapes
: 
s
#batch_normalization_3/cond/switch_fIdentity!batch_normalization_3/cond/Switch*
_output_shapes
: *
T0

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
*batch_normalization_3/cond/batchnorm/RsqrtRsqrt(batch_normalization_3/cond/batchnorm/add*
_output_shapes	
:�*
T0
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
1batch_normalization_3/cond/batchnorm/mul_1/SwitchSwitchdense_3/Relu"batch_normalization_3/cond/pred_id*<
_output_shapes*
(:����������:����������*
T0*
_class
loc:@dense_3/Relu
�
5batch_normalization_3/cond/batchnorm/ReadVariableOp_1ReadVariableOp<batch_normalization_3/cond/batchnorm/ReadVariableOp_1/Switch*
dtype0*
_output_shapes	
:�
�
<batch_normalization_3/cond/batchnorm/ReadVariableOp_1/SwitchSwitch!batch_normalization_3/moving_mean"batch_normalization_3/cond/pred_id*
_output_shapes
: : *
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean
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
dropout_2/cond/switch_tIdentitydropout_2/cond/Switch:1*
_output_shapes
: *
T0

[
dropout_2/cond/switch_fIdentitydropout_2/cond/Switch*
T0
*
_output_shapes
: 
Y
dropout_2/cond/pred_idIdentitykeras_learning_phase*
_output_shapes
: *
T0

z
dropout_2/cond/dropout/rateConst^dropout_2/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *   ?
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
dtype0*
seed2���*(
_output_shapes
:����������*
seed���)
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
%dropout_2/cond/dropout/random_uniformAdd)dropout_2/cond/dropout/random_uniform/mul)dropout_2/cond/dropout/random_uniform/min*
T0*(
_output_shapes
:����������
{
dropout_2/cond/dropout/sub/xConst^dropout_2/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  �?
}
dropout_2/cond/dropout/subSubdropout_2/cond/dropout/sub/xdropout_2/cond/dropout/rate*
_output_shapes
: *
T0
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
dropout_2/cond/dropout/mulMul%dropout_2/cond/dropout/Shape/Switch:1dropout_2/cond/dropout/truediv*
T0*(
_output_shapes
:����������
�
dropout_2/cond/dropout/CastCast#dropout_2/cond/dropout/GreaterEqual*

SrcT0
*
Truncate( *

DstT0*(
_output_shapes
:����������
�
dropout_2/cond/dropout/mul_1Muldropout_2/cond/dropout/muldropout_2/cond/dropout/Cast*(
_output_shapes
:����������*
T0
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
dtype0*
_output_shapes
:*
valueB"�   �   
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
seed���)*
T0*
dtype0*
seed2��h* 
_output_shapes
:
��
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
dense_4/kernelVarHandleOp*
shape:
��*
dtype0*
_output_shapes
: *
shared_namedense_4/kernel*!
_class
loc:@dense_4/kernel*
	container 
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
T0*
transpose_a( *(
_output_shapes
:����������*
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
dense_4/ReluReludense_4/BiasAdd*(
_output_shapes
:����������*
T0
j
batch_normalization_4/ConstConst*
dtype0*
_output_shapes	
:�*
valueB�*  �?
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
batch_normalization_4/Const_1Const*
valueB�*    *
dtype0*
_output_shapes	
:�
�
batch_normalization_4/betaVarHandleOp*
dtype0*
_output_shapes
: *+
shared_namebatch_normalization_4/beta*-
_class#
!loc:@batch_normalization_4/beta*
	container *
shape:�
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
%batch_normalization_4/moving_varianceVarHandleOp*6
shared_name'%batch_normalization_4/moving_variance*8
_class.
,*loc:@batch_normalization_4/moving_variance*
	container *
shape:�*
dtype0*
_output_shapes
: 
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
8batch_normalization_4/moments/variance/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB: 
�
&batch_normalization_4/moments/varianceMean/batch_normalization_4/moments/SquaredDifference8batch_normalization_4/moments/variance/reduction_indices*
T0*
_output_shapes
:	�*

Tidx0*
	keep_dims(
�
%batch_normalization_4/moments/SqueezeSqueeze"batch_normalization_4/moments/mean*
T0*
_output_shapes	
:�*
squeeze_dims
 
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
%batch_normalization_4/batchnorm/RsqrtRsqrt#batch_normalization_4/batchnorm/add*
_output_shapes	
:�*
T0
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
%batch_normalization_4/batchnorm/mul_1Muldense_4/Relu#batch_normalization_4/batchnorm/mul*(
_output_shapes
:����������*
T0
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
+batch_normalization_4/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
#batch_normalization_4/strided_sliceStridedSlicebatch_normalization_4/Shape)batch_normalization_4/strided_slice/stack+batch_normalization_4/strided_slice/stack_1+batch_normalization_4/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
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
batch_normalization_4/sub/yConst*
dtype0*
_output_shapes
: *
valueB
 *� �?
z
batch_normalization_4/subSubbatch_normalization_4/Castbatch_normalization_4/sub/y*
_output_shapes
: *
T0
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
batch_normalization_4/sub_1Sub$batch_normalization_4/ReadVariableOp%batch_normalization_4/moments/Squeeze*
T0*4
_class*
(&loc:@batch_normalization_4/moving_mean*
_output_shapes	
:�
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
&batch_normalization_4/ReadVariableOp_3ReadVariableOp%batch_normalization_4/moving_variance,^batch_normalization_4/AssignSubVariableOp_1*8
_class.
,*loc:@batch_normalization_4/moving_variance*
dtype0*
_output_shapes	
:�
z
!batch_normalization_4/cond/SwitchSwitchkeras_learning_phasekeras_learning_phase*
_output_shapes
: : *
T0

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
*batch_normalization_4/cond/batchnorm/RsqrtRsqrt(batch_normalization_4/cond/batchnorm/add*
_output_shapes	
:�*
T0
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
*batch_normalization_4/cond/batchnorm/add_1AddV2*batch_normalization_4/cond/batchnorm/mul_1(batch_normalization_4/cond/batchnorm/sub*(
_output_shapes
:����������*
T0
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
dtype0*
seed2���*
_output_shapes
:	�+*
seed���)
z
dense_5/random_uniform/subSubdense_5/random_uniform/maxdense_5/random_uniform/min*
_output_shapes
: *
T0
�
dense_5/random_uniform/mulMul$dense_5/random_uniform/RandomUniformdense_5/random_uniform/sub*
T0*
_output_shapes
:	�+

dense_5/random_uniformAdddense_5/random_uniform/muldense_5/random_uniform/min*
T0*
_output_shapes
:	�+
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
,Adam/learning_rate/Initializer/initial_valueConst*%
_class
loc:@Adam/learning_rate*
valueB
 *o�:*
dtype0*
_output_shapes
: 
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
%Adam/beta_1/Initializer/initial_valueConst*
dtype0*
_output_shapes
: *
_class
loc:@Adam/beta_1*
valueB
 *fff?
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
_class
loc:@Adam/decay*
valueB
 *    *
dtype0*
_output_shapes
: 
�

Adam/decayVarHandleOp*
shared_name
Adam/decay*
_class
loc:@Adam/decay*
	container *
shape: *
dtype0*
_output_shapes
: 
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
dense_5_targetPlaceholder*%
shape:������������������*
dtype0*0
_output_shapes
:������������������
q
dense_5_sample_weightsPlaceholder*
dtype0*#
_output_shapes
:���������*
shape:���������
J
ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
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
#metrics/accuracy/ArgMax_1/dimensionConst*
dtype0*
_output_shapes
: *
valueB :
���������
�
metrics/accuracy/ArgMax_1ArgMaxdense_5/Softmax#metrics/accuracy/ArgMax_1/dimension*
T0*
output_type0	*#
_output_shapes
:���������*

Tidx0
�
metrics/accuracy/EqualEqualmetrics/accuracy/ArgMaxmetrics/accuracy/ArgMax_1*#
_output_shapes
:���������*
incompatible_shape_error(*
T0	
�
metrics/accuracy/CastCastmetrics/accuracy/Equal*

SrcT0
*
Truncate( *

DstT0*#
_output_shapes
:���������
`
metrics/accuracy/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
metrics/accuracy/SumSummetrics/accuracy/Castmetrics/accuracy/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
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
metrics/accuracy/truedivRealDiv!metrics/accuracy/ReadVariableOp_2'metrics/accuracy/truediv/ReadVariableOp*
_output_shapes
: *
T0
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
Rloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/ShapeShapedense_5/BiasAdd*
T0*
out_type0*
_output_shapes
:
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
Rloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/SliceSliceTloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Shape_1Xloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Slice/beginWloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Slice/size*
_output_shapes
:*
T0*
Index0
�
\loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/concat/values_0Const*
valueB:
���������*
dtype0*
_output_shapes
:
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
Tloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Slice_1SliceTloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Shape_2Zloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Slice_1/beginYloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Slice_1/size*
T0*
Index0*
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
Zloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
Uloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/concat_1ConcatV2^loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/concat_1/values_0Tloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Slice_1Zloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:
�
Vloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape_1Reshapedense_5_targetUloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/concat_1*0
_output_shapes
:������������������*
T0*
Tshape0
�
Lloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogitsTloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/ReshapeVloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape_1*?
_output_shapes-
+:���������:������������������*
T0
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
Tloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Slice_2SliceRloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/ShapeZloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Slice_2/beginYloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Slice_2/size*
_output_shapes
:*
T0*
Index0
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
loss/mul/x@loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv*
_output_shapes
: *
T0
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
: *

Tidx0*
	keep_dims( 
y
training/Adam/gradients/ShapeConst*
_class
	loc:@Mean*
valueB *
dtype0*
_output_shapes
: 
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
/training/Adam/gradients/Mean_grad/Reshape/shapeConst*
_class
	loc:@Mean*
valueB *
dtype0*
_output_shapes
: 
�
)training/Adam/gradients/Mean_grad/ReshapeReshapetraining/Adam/gradients/Fill/training/Adam/gradients/Mean_grad/Reshape/shape*
T0*
_class
	loc:@Mean*
Tshape0*
_output_shapes
: 
�
'training/Adam/gradients/Mean_grad/ConstConst*
_class
	loc:@Mean*
valueB *
dtype0*
_output_shapes
: 
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
etraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/Shape_1Const*S
_classI
GEloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv*
valueB *
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
etraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/RealDivRealDiv+training/Adam/gradients/loss/mul_grad/Mul_1Jloss/dense_5_loss/categorical_crossentropy/weighted_loss/num_elements/Cast*
T0*S
_classI
GEloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv*
_output_shapes
: 
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
etraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/ReshapeReshapeatraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/Sumctraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/Shape*
T0*S
_classI
GEloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv*
Tshape0*
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
: *
	keep_dims( *

Tidx0
�
gtraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/Reshape_1Reshapectraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/Sum_1etraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv_grad/Shape_1*
_output_shapes
: *
T0*S
_classI
GEloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/truediv*
Tshape0
�
gtraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/Sum_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*O
_classE
CAloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/Sum*
valueB:
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
_training/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul_grad/ShapeShapedense_5_sample_weights*
_output_shapes
:*
T0*O
_classE
CAloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul*
out_type0
�
atraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul_grad/Shape_1ShapeVloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape_2*
_output_shapes
:*
T0*O
_classE
CAloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul*
out_type0
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
atraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul_grad/ReshapeReshape]training/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul_grad/Sum_training/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul_grad/Shape*
T0*O
_classE
CAloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul*
Tshape0*#
_output_shapes
:���������
�
_training/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul_grad/Mul_1Muldense_5_sample_weights^training/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/Sum_grad/Tile*
T0*O
_classE
CAloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul*#
_output_shapes
:���������
�
_training/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul_grad/Sum_1Sum_training/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul_grad/Mul_1qtraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*O
_classE
CAloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul
�
ctraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul_grad/Reshape_1Reshape_training/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul_grad/Sum_1atraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul_grad/Shape_1*
T0*O
_classE
CAloc:@loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul*
Tshape0*#
_output_shapes
:���������
�
ytraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape_2_grad/ShapeShapeLloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits*
T0*i
_class_
][loc:@loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape_2*
out_type0*
_output_shapes
:
�
{training/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape_2_grad/ReshapeReshapectraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/weighted_loss/mul_grad/Reshape_1ytraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape_2_grad/Shape*
T0*i
_class_
][loc:@loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape_2*
Tshape0*#
_output_shapes
:���������
�
"training/Adam/gradients/zeros_like	ZerosLikeNloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits:1*0
_output_shapes
:������������������*
T0*_
_classU
SQloc:@loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits
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
ExpandDims{training/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape_2_grad/Reshapextraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits_grad/ExpandDims/dim*

Tdim0*
T0*_
_classU
SQloc:@loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits*'
_output_shapes
:���������
�
mtraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits_grad/mulMulttraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits_grad/ExpandDimsNloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits:1*
T0*_
_classU
SQloc:@loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits*0
_output_shapes
:������������������
�
ttraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits_grad/LogSoftmax
LogSoftmaxTloss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape*
T0*_
_classU
SQloc:@loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits*0
_output_shapes
:������������������
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
ExpandDims{training/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape_2_grad/Reshapeztraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits_grad/ExpandDims_1/dim*

Tdim0*
T0*_
_classU
SQloc:@loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits*'
_output_shapes
:���������
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
ytraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape_grad/ReshapeReshapemtraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits_grad/mulwtraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape_grad/Shape*
T0*g
_class]
[Yloc:@loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape*
Tshape0*'
_output_shapes
:���������+
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
4training/Adam/gradients/dense_5/MatMul_grad/MatMul_1MatMul batch_normalization_4/cond/Mergeytraining/Adam/gradients/loss/dense_5_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape_grad/Reshape*
T0*!
_class
loc:@dense_5/MatMul*
transpose_a(*
_output_shapes
:	�+*
transpose_b( 
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
Otraining/Adam/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/Shape_1Shape(batch_normalization_4/cond/batchnorm/sub*
T0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/add_1*
out_type0*
_output_shapes
:
�
]training/Adam/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsMtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/ShapeOtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/Shape_1*
T0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/add_1*2
_output_shapes 
:���������:���������
�
Ktraining/Adam/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/SumSumGtraining/Adam/gradients/batch_normalization_4/cond/Merge_grad/cond_grad]training/Adam/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/BroadcastGradientArgs*
T0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/add_1*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Otraining/Adam/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/ReshapeReshapeKtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/SumMtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/Shape*
T0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/add_1*
Tshape0*(
_output_shapes
:����������
�
Mtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/Sum_1SumGtraining/Adam/gradients/batch_normalization_4/cond/Merge_grad/cond_grad_training/Adam/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/add_1*
_output_shapes
:*
	keep_dims( *

Tidx0
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
 training/Adam/gradients/IdentityIdentitytraining/Adam/gradients/Switch*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1*(
_output_shapes
:����������
�
training/Adam/gradients/Shape_1Shapetraining/Adam/gradients/Switch*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1*
out_type0*
_output_shapes
:
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
Otraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/Shape_1Shape(batch_normalization_4/cond/batchnorm/mul*
T0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/mul_1*
out_type0*
_output_shapes
:
�
]training/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsMtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/ShapeOtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/Shape_1*
T0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/mul_1*2
_output_shapes 
:���������:���������
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
Mtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/Sum_1SumMtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/Mul_1_training/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
T0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/mul_1*
_output_shapes
:*
	keep_dims( *

Tidx0
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
Htraining/Adam/gradients/batch_normalization_4/batchnorm/add_1_grad/ShapeShape%batch_normalization_4/batchnorm/mul_1*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1*
out_type0*
_output_shapes
:
�
Jtraining/Adam/gradients/batch_normalization_4/batchnorm/add_1_grad/Shape_1Shape#batch_normalization_4/batchnorm/sub*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1*
out_type0*
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
Ftraining/Adam/gradients/batch_normalization_4/batchnorm/add_1_grad/SumSumJtraining/Adam/gradients/batch_normalization_4/cond/Switch_1_grad/cond_gradXtraining/Adam/gradients/batch_normalization_4/batchnorm/add_1_grad/BroadcastGradientArgs*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Jtraining/Adam/gradients/batch_normalization_4/batchnorm/add_1_grad/ReshapeReshapeFtraining/Adam/gradients/batch_normalization_4/batchnorm/add_1_grad/SumHtraining/Adam/gradients/batch_normalization_4/batchnorm/add_1_grad/Shape*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1*
Tshape0*(
_output_shapes
:����������
�
Htraining/Adam/gradients/batch_normalization_4/batchnorm/add_1_grad/Sum_1SumJtraining/Adam/gradients/batch_normalization_4/cond/Switch_1_grad/cond_gradZtraining/Adam/gradients/batch_normalization_4/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Ltraining/Adam/gradients/batch_normalization_4/batchnorm/add_1_grad/Reshape_1ReshapeHtraining/Adam/gradients/batch_normalization_4/batchnorm/add_1_grad/Sum_1Jtraining/Adam/gradients/batch_normalization_4/batchnorm/add_1_grad/Shape_1*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1*
Tshape0*
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
N**
_output_shapes
:����������: *
T0*
_class
loc:@dense_4/Relu
�
Ktraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_2_grad/MulMulItraining/Adam/gradients/batch_normalization_4/cond/batchnorm/sub_grad/Neg(batch_normalization_4/cond/batchnorm/mul*
_output_shapes	
:�*
T0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/mul_2
�
Mtraining/Adam/gradients/batch_normalization_4/cond/batchnorm/mul_2_grad/Mul_1MulItraining/Adam/gradients/batch_normalization_4/cond/batchnorm/sub_grad/Neg5batch_normalization_4/cond/batchnorm/ReadVariableOp_1*
_output_shapes	
:�*
T0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/mul_2
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
Ftraining/Adam/gradients/batch_normalization_4/batchnorm/mul_1_grad/MulMulJtraining/Adam/gradients/batch_normalization_4/batchnorm/add_1_grad/Reshape#batch_normalization_4/batchnorm/mul*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/mul_1*(
_output_shapes
:����������
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
Htraining/Adam/gradients/batch_normalization_4/batchnorm/mul_1_grad/Mul_1Muldense_4/ReluJtraining/Adam/gradients/batch_normalization_4/batchnorm/add_1_grad/Reshape*(
_output_shapes
:����������*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/mul_1
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
Htraining/Adam/gradients/batch_normalization_4/batchnorm/mul_2_grad/Mul_1MulDtraining/Adam/gradients/batch_normalization_4/batchnorm/sub_grad/Neg%batch_normalization_4/moments/Squeeze*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/mul_2*
_output_shapes	
:�
�
training/Adam/gradients/AddN_1AddNctraining/Adam/gradients/batch_normalization_4/cond/batchnorm/ReadVariableOp_2/Switch_grad/cond_gradLtraining/Adam/gradients/batch_normalization_4/batchnorm/add_1_grad/Reshape_1*
N*
_output_shapes	
:�*
T0*-
_class#
!loc:@batch_normalization_4/beta
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
'training/Adam/gradients/VariableShape_1VariableShape"training/Adam/gradients/Switch_3:1#^training/Adam/gradients/Identity_3*.
_class$
" loc:@batch_normalization_4/gamma*
out_type0*
_output_shapes
:
�
%training/Adam/gradients/zeros_3/ConstConst#^training/Adam/gradients/Identity_3*.
_class$
" loc:@batch_normalization_4/gamma*
valueB
 *    *
dtype0*
_output_shapes
: 
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
T0*.
_class$
" loc:@batch_normalization_4/gamma*
N*%
_output_shapes
:���������: 
�
Ltraining/Adam/gradients/batch_normalization_4/batchnorm/Rsqrt_grad/RsqrtGrad	RsqrtGrad%batch_normalization_4/batchnorm/RsqrtDtraining/Adam/gradients/batch_normalization_4/batchnorm/mul_grad/Mul*
_output_shapes	
:�*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/Rsqrt
�
Ytraining/Adam/gradients/batch_normalization_4/batchnorm/add_grad/BroadcastGradientArgs/s0Const*
dtype0*
_output_shapes
:*6
_class,
*(loc:@batch_normalization_4/batchnorm/add*
valueB:�
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
Ntraining/Adam/gradients/batch_normalization_4/batchnorm/add_grad/Reshape/shapeConst*6
_class,
*(loc:@batch_normalization_4/batchnorm/add*
valueB *
dtype0*
_output_shapes
: 
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
N*
_output_shapes	
:�*
T0*.
_class$
" loc:@batch_normalization_4/gamma
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
Ktraining/Adam/gradients/batch_normalization_4/moments/variance_grad/Shape_1Const*
dtype0*
_output_shapes
:*9
_class/
-+loc:@batch_normalization_4/moments/variance*
valueB:
�
Otraining/Adam/gradients/batch_normalization_4/moments/variance_grad/range/startConst*
dtype0*
_output_shapes
: *9
_class/
-+loc:@batch_normalization_4/moments/variance*
value	B : 
�
Otraining/Adam/gradients/batch_normalization_4/moments/variance_grad/range/deltaConst*
dtype0*
_output_shapes
: *9
_class/
-+loc:@batch_normalization_4/moments/variance*
value	B :
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
T0*9
_class/
-+loc:@batch_normalization_4/moments/variance*
N*
_output_shapes
:
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
T0*9
_class/
-+loc:@batch_normalization_4/moments/variance*
out_type0
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
Jtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/Prod_1ProdKtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/Shape_3Ktraining/Adam/gradients/batch_normalization_4/moments/variance_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0*9
_class/
-+loc:@batch_normalization_4/moments/variance
�
Otraining/Adam/gradients/batch_normalization_4/moments/variance_grad/Maximum_1/yConst*9
_class/
-+loc:@batch_normalization_4/moments/variance*
value	B :*
dtype0*
_output_shapes
: 
�
Mtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/Maximum_1MaximumJtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/Prod_1Otraining/Adam/gradients/batch_normalization_4/moments/variance_grad/Maximum_1/y*
_output_shapes
: *
T0*9
_class/
-+loc:@batch_normalization_4/moments/variance
�
Ntraining/Adam/gradients/batch_normalization_4/moments/variance_grad/floordiv_1FloorDivHtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/ProdMtraining/Adam/gradients/batch_normalization_4/moments/variance_grad/Maximum_1*
T0*9
_class/
-+loc:@batch_normalization_4/moments/variance*
_output_shapes
: 
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
T0*B
_class8
64loc:@batch_normalization_4/moments/SquaredDifference*
out_type0*
_output_shapes
:
�
Ttraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/Shape_1Shape*batch_normalization_4/moments/StopGradient*
T0*B
_class8
64loc:@batch_normalization_4/moments/SquaredDifference*
out_type0*
_output_shapes
:
�
btraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgsRtraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/ShapeTtraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/Shape_1*
T0*B
_class8
64loc:@batch_normalization_4/moments/SquaredDifference*2
_output_shapes 
:���������:���������
�
Ptraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/SumSumRtraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/mul_1btraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*B
_class8
64loc:@batch_normalization_4/moments/SquaredDifference*
_output_shapes
:
�
Ttraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/ReshapeReshapePtraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/SumRtraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/Shape*
T0*B
_class8
64loc:@batch_normalization_4/moments/SquaredDifference*
Tshape0*(
_output_shapes
:����������
�
Rtraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/Sum_1SumRtraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/mul_1dtraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/BroadcastGradientArgs:1*
T0*B
_class8
64loc:@batch_normalization_4/moments/SquaredDifference*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Vtraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/Reshape_1ReshapeRtraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/Sum_1Ttraining/Adam/gradients/batch_normalization_4/moments/SquaredDifference_grad/Shape_1*
T0*B
_class8
64loc:@batch_normalization_4/moments/SquaredDifference*
Tshape0*
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
Ctraining/Adam/gradients/batch_normalization_4/moments/mean_grad/addAddV24batch_normalization_4/moments/mean/reduction_indicesDtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Size*
T0*5
_class+
)'loc:@batch_normalization_4/moments/mean*
_output_shapes
:
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
N*
_output_shapes
:*
T0*5
_class+
)'loc:@batch_normalization_4/moments/mean
�
Itraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Maximum/yConst*
dtype0*
_output_shapes
: *5
_class+
)'loc:@batch_normalization_4/moments/mean*
value	B :
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
T0*5
_class+
)'loc:@batch_normalization_4/moments/mean*
Tshape0*0
_output_shapes
:������������������
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
Dtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/ProdProdGtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Shape_2Etraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0*5
_class+
)'loc:@batch_normalization_4/moments/mean
�
Gtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Const_1Const*5
_class+
)'loc:@batch_normalization_4/moments/mean*
valueB: *
dtype0*
_output_shapes
:
�
Ftraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Prod_1ProdGtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Shape_3Gtraining/Adam/gradients/batch_normalization_4/moments/mean_grad/Const_1*
T0*5
_class+
)'loc:@batch_normalization_4/moments/mean*
_output_shapes
: *
	keep_dims( *

Tidx0
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
2training/Adam/gradients/dense_4/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_4/Relu_grad/ReluGraddense_4/MatMul/ReadVariableOp*
transpose_a( *(
_output_shapes
:����������*
transpose_b(*
T0*!
_class
loc:@dense_4/MatMul
�
4training/Adam/gradients/dense_4/MatMul_grad/MatMul_1MatMuldropout_2/cond/Merge2training/Adam/gradients/dense_4/Relu_grad/ReluGrad*
T0*!
_class
loc:@dense_4/MatMul*
transpose_a(* 
_output_shapes
:
��*
transpose_b( 
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
%training/Adam/gradients/zeros_4/ConstConst#^training/Adam/gradients/Identity_4*
dtype0*
_output_shapes
: *3
_class)
'%loc:@batch_normalization_3/cond/Merge*
valueB
 *    
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
T0*3
_class)
'%loc:@batch_normalization_3/cond/Merge*
N**
_output_shapes
:����������: 
�
?training/Adam/gradients/dropout_2/cond/dropout/mul_1_grad/ShapeShapedropout_2/cond/dropout/mul*
_output_shapes
:*
T0*/
_class%
#!loc:@dropout_2/cond/dropout/mul_1*
out_type0
�
Atraining/Adam/gradients/dropout_2/cond/dropout/mul_1_grad/Shape_1Shapedropout_2/cond/dropout/Cast*
_output_shapes
:*
T0*/
_class%
#!loc:@dropout_2/cond/dropout/mul_1*
out_type0
�
Otraining/Adam/gradients/dropout_2/cond/dropout/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs?training/Adam/gradients/dropout_2/cond/dropout/mul_1_grad/ShapeAtraining/Adam/gradients/dropout_2/cond/dropout/mul_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0*/
_class%
#!loc:@dropout_2/cond/dropout/mul_1
�
=training/Adam/gradients/dropout_2/cond/dropout/mul_1_grad/MulMul=training/Adam/gradients/dropout_2/cond/Merge_grad/cond_grad:1dropout_2/cond/dropout/Cast*
T0*/
_class%
#!loc:@dropout_2/cond/dropout/mul_1*(
_output_shapes
:����������
�
=training/Adam/gradients/dropout_2/cond/dropout/mul_1_grad/SumSum=training/Adam/gradients/dropout_2/cond/dropout/mul_1_grad/MulOtraining/Adam/gradients/dropout_2/cond/dropout/mul_1_grad/BroadcastGradientArgs*
T0*/
_class%
#!loc:@dropout_2/cond/dropout/mul_1*
_output_shapes
:*
	keep_dims( *

Tidx0
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
=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/ShapeShape%dropout_2/cond/dropout/Shape/Switch:1*
_output_shapes
:*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul*
out_type0
�
?training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Shape_1Shapedropout_2/cond/dropout/truediv*
_output_shapes
: *
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul*
out_type0
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
;training/Adam/gradients/dropout_2/cond/dropout/mul_grad/SumSum;training/Adam/gradients/dropout_2/cond/dropout/mul_grad/MulMtraining/Adam/gradients/dropout_2/cond/dropout/mul_grad/BroadcastGradientArgs*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul*
_output_shapes
:*
	keep_dims( *

Tidx0
�
?training/Adam/gradients/dropout_2/cond/dropout/mul_grad/ReshapeReshape;training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Sum=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Shape*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul*
Tshape0*(
_output_shapes
:����������
�
=training/Adam/gradients/dropout_2/cond/dropout/mul_grad/Mul_1Mul%dropout_2/cond/dropout/Shape/Switch:1Atraining/Adam/gradients/dropout_2/cond/dropout/mul_1_grad/Reshape*
T0*-
_class#
!loc:@dropout_2/cond/dropout/mul*(
_output_shapes
:����������
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
"training/Adam/gradients/Identity_5Identity training/Adam/gradients/Switch_5*
T0*3
_class)
'%loc:@batch_normalization_3/cond/Merge*(
_output_shapes
:����������
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
N**
_output_shapes
:����������: *
T0*3
_class)
'%loc:@batch_normalization_3/cond/Merge
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
T0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/add_1*
out_type0*
_output_shapes
:
�
Otraining/Adam/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/Shape_1Shape(batch_normalization_3/cond/batchnorm/sub*
T0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/add_1*
out_type0*
_output_shapes
:
�
]training/Adam/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsMtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/ShapeOtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/Shape_1*
T0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/add_1*2
_output_shapes 
:���������:���������
�
Ktraining/Adam/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/SumSumGtraining/Adam/gradients/batch_normalization_3/cond/Merge_grad/cond_grad]training/Adam/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/BroadcastGradientArgs*
T0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/add_1*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Otraining/Adam/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/ReshapeReshapeKtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/SumMtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/Shape*
T0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/add_1*
Tshape0*(
_output_shapes
:����������
�
Mtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/Sum_1SumGtraining/Adam/gradients/batch_normalization_3/cond/Merge_grad/cond_grad_training/Adam/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/add_1
�
Qtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/Reshape_1ReshapeMtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/Sum_1Otraining/Adam/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/Shape_1*
T0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/add_1*
Tshape0*
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
"training/Adam/gradients/Identity_6Identity training/Adam/gradients/Switch_6*(
_output_shapes
:����������*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1
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
Ktraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/SumSumKtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/Mul]training/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/mul_1*
_output_shapes
:
�
Otraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/ReshapeReshapeKtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/SumMtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/Shape*(
_output_shapes
:����������*
T0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/mul_1*
Tshape0
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
:*
	keep_dims( *

Tidx0
�
Qtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/Reshape_1ReshapeMtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/Sum_1Otraining/Adam/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/Shape_1*
T0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/mul_1*
Tshape0*
_output_shapes	
:�
�
Itraining/Adam/gradients/batch_normalization_3/cond/batchnorm/sub_grad/NegNegQtraining/Adam/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:�*
T0*;
_class1
/-loc:@batch_normalization_3/cond/batchnorm/sub
�
Htraining/Adam/gradients/batch_normalization_3/batchnorm/add_1_grad/ShapeShape%batch_normalization_3/batchnorm/mul_1*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1*
out_type0*
_output_shapes
:
�
Jtraining/Adam/gradients/batch_normalization_3/batchnorm/add_1_grad/Shape_1Shape#batch_normalization_3/batchnorm/sub*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1*
out_type0*
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
Ftraining/Adam/gradients/batch_normalization_3/batchnorm/add_1_grad/SumSumJtraining/Adam/gradients/batch_normalization_3/cond/Switch_1_grad/cond_gradXtraining/Adam/gradients/batch_normalization_3/batchnorm/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1
�
Jtraining/Adam/gradients/batch_normalization_3/batchnorm/add_1_grad/ReshapeReshapeFtraining/Adam/gradients/batch_normalization_3/batchnorm/add_1_grad/SumHtraining/Adam/gradients/batch_normalization_3/batchnorm/add_1_grad/Shape*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1*
Tshape0*(
_output_shapes
:����������
�
Htraining/Adam/gradients/batch_normalization_3/batchnorm/add_1_grad/Sum_1SumJtraining/Adam/gradients/batch_normalization_3/cond/Switch_1_grad/cond_gradZtraining/Adam/gradients/batch_normalization_3/batchnorm/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1*
_output_shapes
:
�
Ltraining/Adam/gradients/batch_normalization_3/batchnorm/add_1_grad/Reshape_1ReshapeHtraining/Adam/gradients/batch_normalization_3/batchnorm/add_1_grad/Sum_1Jtraining/Adam/gradients/batch_normalization_3/batchnorm/add_1_grad/Shape_1*
_output_shapes	
:�*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1*
Tshape0
�
 training/Adam/gradients/Switch_7Switchdense_3/Relu"batch_normalization_3/cond/pred_id*<
_output_shapes*
(:����������:����������*
T0*
_class
loc:@dense_3/Relu
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
dtype0*
_output_shapes
: *
_class
loc:@dense_3/Relu*
valueB
 *    
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
Htraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/ShapeShapedense_3/Relu*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/mul_1*
out_type0*
_output_shapes
:
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
Ftraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/SumSumFtraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/MulXtraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/mul_1
�
Jtraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/ReshapeReshapeFtraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/SumHtraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/Shape*(
_output_shapes
:����������*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/mul_1*
Tshape0
�
Htraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/Mul_1Muldense_3/ReluJtraining/Adam/gradients/batch_normalization_3/batchnorm/add_1_grad/Reshape*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/mul_1*(
_output_shapes
:����������
�
Htraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/Sum_1SumHtraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/Mul_1Ztraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/mul_1*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Ltraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/Reshape_1ReshapeHtraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/Sum_1Jtraining/Adam/gradients/batch_normalization_3/batchnorm/mul_1_grad/Shape_1*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/mul_1*
Tshape0*
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
'training/Adam/gradients/VariableShape_2VariableShape"training/Adam/gradients/Switch_8:1#^training/Adam/gradients/Identity_8*
_output_shapes
:*-
_class#
!loc:@batch_normalization_3/beta*
out_type0
�
%training/Adam/gradients/zeros_8/ConstConst#^training/Adam/gradients/Identity_8*-
_class#
!loc:@batch_normalization_3/beta*
valueB
 *    *
dtype0*
_output_shapes
: 
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
T0*-
_class#
!loc:@batch_normalization_3/beta*
N*
_output_shapes	
:�
�
Htraining/Adam/gradients/batch_normalization_3/moments/Squeeze_grad/ShapeConst*8
_class.
,*loc:@batch_normalization_3/moments/Squeeze*
valueB"   �   *
dtype0*
_output_shapes
:
�
Jtraining/Adam/gradients/batch_normalization_3/moments/Squeeze_grad/ReshapeReshapeFtraining/Adam/gradients/batch_normalization_3/batchnorm/mul_2_grad/MulHtraining/Adam/gradients/batch_normalization_3/moments/Squeeze_grad/Shape*
_output_shapes
:	�*
T0*8
_class.
,*loc:@batch_normalization_3/moments/Squeeze*
Tshape0
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
Ftraining/Adam/gradients/batch_normalization_3/batchnorm/mul_grad/Mul_1Multraining/Adam/gradients/AddN_8%batch_normalization_3/batchnorm/Rsqrt*
T0*6
_class,
*(loc:@batch_normalization_3/batchnorm/mul*
_output_shapes	
:�
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
'training/Adam/gradients/VariableShape_3VariableShape"training/Adam/gradients/Switch_9:1#^training/Adam/gradients/Identity_9*.
_class$
" loc:@batch_normalization_3/gamma*
out_type0*
_output_shapes
:
�
%training/Adam/gradients/zeros_9/ConstConst#^training/Adam/gradients/Identity_9*
dtype0*
_output_shapes
: *.
_class$
" loc:@batch_normalization_3/gamma*
valueB
 *    
�
training/Adam/gradients/zeros_9Fill'training/Adam/gradients/VariableShape_3%training/Adam/gradients/zeros_9/Const*
T0*.
_class$
" loc:@batch_normalization_3/gamma*

index_type0*#
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
Vtraining/Adam/gradients/batch_normalization_3/batchnorm/add_grad/Sum/reduction_indicesConst*6
_class,
*(loc:@batch_normalization_3/batchnorm/add*
valueB: *
dtype0*
_output_shapes
:
�
Dtraining/Adam/gradients/batch_normalization_3/batchnorm/add_grad/SumSumLtraining/Adam/gradients/batch_normalization_3/batchnorm/Rsqrt_grad/RsqrtGradVtraining/Adam/gradients/batch_normalization_3/batchnorm/add_grad/Sum/reduction_indices*
T0*6
_class,
*(loc:@batch_normalization_3/batchnorm/add*
_output_shapes
: *
	keep_dims( *

Tidx0
�
Ntraining/Adam/gradients/batch_normalization_3/batchnorm/add_grad/Reshape/shapeConst*
dtype0*
_output_shapes
: *6
_class,
*(loc:@batch_normalization_3/batchnorm/add*
valueB 
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
T0*.
_class$
" loc:@batch_normalization_3/gamma*
N*
_output_shapes	
:�
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
Itraining/Adam/gradients/batch_normalization_3/moments/variance_grad/ShapeShape/batch_normalization_3/moments/SquaredDifference*
T0*9
_class/
-+loc:@batch_normalization_3/moments/variance*
out_type0*
_output_shapes
:
�
Htraining/Adam/gradients/batch_normalization_3/moments/variance_grad/SizeConst*
dtype0*
_output_shapes
: *9
_class/
-+loc:@batch_normalization_3/moments/variance*
value	B :
�
Gtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/addAddV28batch_normalization_3/moments/variance/reduction_indicesHtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Size*
_output_shapes
:*
T0*9
_class/
-+loc:@batch_normalization_3/moments/variance
�
Gtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/modFloorModGtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/addHtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Size*
T0*9
_class/
-+loc:@batch_normalization_3/moments/variance*
_output_shapes
:
�
Ktraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Shape_1Const*
dtype0*
_output_shapes
:*9
_class/
-+loc:@batch_normalization_3/moments/variance*
valueB:
�
Otraining/Adam/gradients/batch_normalization_3/moments/variance_grad/range/startConst*9
_class/
-+loc:@batch_normalization_3/moments/variance*
value	B : *
dtype0*
_output_shapes
: 
�
Otraining/Adam/gradients/batch_normalization_3/moments/variance_grad/range/deltaConst*9
_class/
-+loc:@batch_normalization_3/moments/variance*
value	B :*
dtype0*
_output_shapes
: 
�
Itraining/Adam/gradients/batch_normalization_3/moments/variance_grad/rangeRangeOtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/range/startHtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/SizeOtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/range/delta*9
_class/
-+loc:@batch_normalization_3/moments/variance*
_output_shapes
:*

Tidx0
�
Ntraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Fill/valueConst*9
_class/
-+loc:@batch_normalization_3/moments/variance*
value	B :*
dtype0*
_output_shapes
: 
�
Htraining/Adam/gradients/batch_normalization_3/moments/variance_grad/FillFillKtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Shape_1Ntraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Fill/value*
_output_shapes
:*
T0*9
_class/
-+loc:@batch_normalization_3/moments/variance*

index_type0
�
Qtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/DynamicStitchDynamicStitchItraining/Adam/gradients/batch_normalization_3/moments/variance_grad/rangeGtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/modItraining/Adam/gradients/batch_normalization_3/moments/variance_grad/ShapeHtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Fill*
T0*9
_class/
-+loc:@batch_normalization_3/moments/variance*
N*
_output_shapes
:
�
Mtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Maximum/yConst*
dtype0*
_output_shapes
: *9
_class/
-+loc:@batch_normalization_3/moments/variance*
value	B :
�
Ktraining/Adam/gradients/batch_normalization_3/moments/variance_grad/MaximumMaximumQtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/DynamicStitchMtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Maximum/y*
T0*9
_class/
-+loc:@batch_normalization_3/moments/variance*
_output_shapes
:
�
Ltraining/Adam/gradients/batch_normalization_3/moments/variance_grad/floordivFloorDivItraining/Adam/gradients/batch_normalization_3/moments/variance_grad/ShapeKtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Maximum*
_output_shapes
:*
T0*9
_class/
-+loc:@batch_normalization_3/moments/variance
�
Ktraining/Adam/gradients/batch_normalization_3/moments/variance_grad/ReshapeReshapeLtraining/Adam/gradients/batch_normalization_3/moments/Squeeze_1_grad/ReshapeQtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/DynamicStitch*
T0*9
_class/
-+loc:@batch_normalization_3/moments/variance*
Tshape0*0
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
Ktraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Const_1Const*
dtype0*
_output_shapes
:*9
_class/
-+loc:@batch_normalization_3/moments/variance*
valueB: 
�
Jtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Prod_1ProdKtraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Shape_3Ktraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Const_1*
T0*9
_class/
-+loc:@batch_normalization_3/moments/variance*
_output_shapes
: *
	keep_dims( *

Tidx0
�
Otraining/Adam/gradients/batch_normalization_3/moments/variance_grad/Maximum_1/yConst*
dtype0*
_output_shapes
: *9
_class/
-+loc:@batch_normalization_3/moments/variance*
value	B :
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
Straining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/scalarConstL^training/Adam/gradients/batch_normalization_3/moments/variance_grad/truediv*B
_class8
64loc:@batch_normalization_3/moments/SquaredDifference*
valueB
 *   @*
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
Ptraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/subSubdense_3/Relu*batch_normalization_3/moments/StopGradientL^training/Adam/gradients/batch_normalization_3/moments/variance_grad/truediv*(
_output_shapes
:����������*
T0*B
_class8
64loc:@batch_normalization_3/moments/SquaredDifference
�
Rtraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/mul_1MulPtraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/MulPtraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/sub*
T0*B
_class8
64loc:@batch_normalization_3/moments/SquaredDifference*(
_output_shapes
:����������
�
Rtraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/ShapeShapedense_3/Relu*
_output_shapes
:*
T0*B
_class8
64loc:@batch_normalization_3/moments/SquaredDifference*
out_type0
�
Ttraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/Shape_1Shape*batch_normalization_3/moments/StopGradient*
T0*B
_class8
64loc:@batch_normalization_3/moments/SquaredDifference*
out_type0*
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
Rtraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/Sum_1SumRtraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/mul_1dtraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/BroadcastGradientArgs:1*
T0*B
_class8
64loc:@batch_normalization_3/moments/SquaredDifference*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Vtraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/Reshape_1ReshapeRtraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/Sum_1Ttraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/Shape_1*
T0*B
_class8
64loc:@batch_normalization_3/moments/SquaredDifference*
Tshape0*
_output_shapes
:	�
�
Ptraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/NegNegVtraining/Adam/gradients/batch_normalization_3/moments/SquaredDifference_grad/Reshape_1*
_output_shapes
:	�*
T0*B
_class8
64loc:@batch_normalization_3/moments/SquaredDifference
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
Ktraining/Adam/gradients/batch_normalization_3/moments/mean_grad/range/deltaConst*5
_class+
)'loc:@batch_normalization_3/moments/mean*
value	B :*
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
Jtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/Fill/valueConst*
dtype0*
_output_shapes
: *5
_class+
)'loc:@batch_normalization_3/moments/mean*
value	B :
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
Htraining/Adam/gradients/batch_normalization_3/moments/mean_grad/floordivFloorDivEtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/ShapeGtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/Maximum*
T0*5
_class+
)'loc:@batch_normalization_3/moments/mean*
_output_shapes
:
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
Dtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/ProdProdGtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/Shape_2Etraining/Adam/gradients/batch_normalization_3/moments/mean_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0*5
_class+
)'loc:@batch_normalization_3/moments/mean
�
Gtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/Const_1Const*5
_class+
)'loc:@batch_normalization_3/moments/mean*
valueB: *
dtype0*
_output_shapes
:
�
Ftraining/Adam/gradients/batch_normalization_3/moments/mean_grad/Prod_1ProdGtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/Shape_3Gtraining/Adam/gradients/batch_normalization_3/moments/mean_grad/Const_1*
T0*5
_class+
)'loc:@batch_normalization_3/moments/mean*
_output_shapes
: *
	keep_dims( *

Tidx0
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
loc:@dense_3/MatMul*
transpose_a( *(
_output_shapes
:����������
�
4training/Adam/gradients/dense_3/MatMul_grad/MatMul_1MatMuldropout_1/cond/Merge2training/Adam/gradients/dense_3/Relu_grad/ReluGrad*
transpose_a(* 
_output_shapes
:
��*
transpose_b( *
T0*!
_class
loc:@dense_3/MatMul
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
 training/Adam/gradients/zeros_10Filltraining/Adam/gradients/Shape_7&training/Adam/gradients/zeros_10/Const*(
_output_shapes
:����������*
T0*3
_class)
'%loc:@batch_normalization_2/cond/Merge*

index_type0
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
:*
	keep_dims( *

Tidx0
�
Atraining/Adam/gradients/dropout_1/cond/dropout/mul_1_grad/ReshapeReshape=training/Adam/gradients/dropout_1/cond/dropout/mul_1_grad/Sum?training/Adam/gradients/dropout_1/cond/dropout/mul_1_grad/Shape*(
_output_shapes
:����������*
T0*/
_class%
#!loc:@dropout_1/cond/dropout/mul_1*
Tshape0
�
?training/Adam/gradients/dropout_1/cond/dropout/mul_1_grad/Mul_1Muldropout_1/cond/dropout/mul=training/Adam/gradients/dropout_1/cond/Merge_grad/cond_grad:1*
T0*/
_class%
#!loc:@dropout_1/cond/dropout/mul_1*(
_output_shapes
:����������
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
=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/ShapeShape%dropout_1/cond/dropout/Shape/Switch:1*
_output_shapes
:*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*
out_type0
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
?training/Adam/gradients/dropout_1/cond/dropout/mul_grad/ReshapeReshape;training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Sum=training/Adam/gradients/dropout_1/cond/dropout/mul_grad/Shape*(
_output_shapes
:����������*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*
Tshape0
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
:*
	keep_dims( *

Tidx0
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
&training/Adam/gradients/zeros_11/ConstConst$^training/Adam/gradients/Identity_11*
dtype0*
_output_shapes
: *3
_class)
'%loc:@batch_normalization_2/cond/Merge*
valueB
 *    
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
N**
_output_shapes
:����������: *
T0*3
_class)
'%loc:@batch_normalization_2/cond/Merge
�
training/Adam/gradients/AddN_11AddN>training/Adam/gradients/dropout_1/cond/Switch_1_grad/cond_gradJtraining/Adam/gradients/dropout_1/cond/dropout/Shape/Switch_grad/cond_grad*
T0*3
_class)
'%loc:@batch_normalization_2/cond/Merge*
N*(
_output_shapes
:����������
�
Gtraining/Adam/gradients/batch_normalization_2/cond/Merge_grad/cond_gradSwitchtraining/Adam/gradients/AddN_11"batch_normalization_2/cond/pred_id*<
_output_shapes*
(:����������:����������*
T0*3
_class)
'%loc:@batch_normalization_2/cond/Merge
�
Mtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/ShapeShape*batch_normalization_2/cond/batchnorm/mul_1*
T0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/add_1*
out_type0*
_output_shapes
:
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
Ktraining/Adam/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/SumSumGtraining/Adam/gradients/batch_normalization_2/cond/Merge_grad/cond_grad]training/Adam/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/add_1*
_output_shapes
:
�
Otraining/Adam/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/ReshapeReshapeKtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/SumMtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/Shape*
T0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/add_1*
Tshape0*(
_output_shapes
:����������
�
Mtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/Sum_1SumGtraining/Adam/gradients/batch_normalization_2/cond/Merge_grad/cond_grad_training/Adam/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/add_1*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Qtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/Reshape_1ReshapeMtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/Sum_1Otraining/Adam/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/Shape_1*
T0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/add_1*
Tshape0*
_output_shapes	
:�
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
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/add_1*
N**
_output_shapes
:����������: 
�
Mtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/ShapeShape1batch_normalization_2/cond/batchnorm/mul_1/Switch*
T0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/mul_1*
out_type0*
_output_shapes
:
�
Otraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/Shape_1Shape(batch_normalization_2/cond/batchnorm/mul*
_output_shapes
:*
T0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/mul_1*
out_type0
�
]training/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsMtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/ShapeOtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/mul_1
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
Otraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/ReshapeReshapeKtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/SumMtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/Shape*
T0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/mul_1*
Tshape0*(
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
Mtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/Sum_1SumMtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/Mul_1_training/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
T0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/mul_1*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Qtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/Reshape_1ReshapeMtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/Sum_1Otraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/Shape_1*
_output_shapes	
:�*
T0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/mul_1*
Tshape0
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
Htraining/Adam/gradients/batch_normalization_2/batchnorm/add_1_grad/Sum_1SumJtraining/Adam/gradients/batch_normalization_2/cond/Switch_1_grad/cond_gradZtraining/Adam/gradients/batch_normalization_2/batchnorm/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/add_1
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
#training/Adam/gradients/Identity_13Identity#training/Adam/gradients/Switch_13:1*
T0*
_class
loc:@dense_2/Relu*(
_output_shapes
:����������
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
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/mul_1*
out_type0*
_output_shapes
:
�
Jtraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/Shape_1Shape#batch_normalization_2/batchnorm/mul*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/mul_1*
out_type0*
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
Ftraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/MulMulJtraining/Adam/gradients/batch_normalization_2/batchnorm/add_1_grad/Reshape#batch_normalization_2/batchnorm/mul*(
_output_shapes
:����������*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/mul_1
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
Jtraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/ReshapeReshapeFtraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/SumHtraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/Shape*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/mul_1*
Tshape0*(
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
Htraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/Sum_1SumHtraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/Mul_1Ztraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/mul_1*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Ltraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/Reshape_1ReshapeHtraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/Sum_1Jtraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/Shape_1*
_output_shapes	
:�*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/mul_1*
Tshape0
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
'training/Adam/gradients/VariableShape_4VariableShape#training/Adam/gradients/Switch_14:1$^training/Adam/gradients/Identity_14*-
_class#
!loc:@batch_normalization_2/beta*
out_type0*
_output_shapes
:
�
&training/Adam/gradients/zeros_14/ConstConst$^training/Adam/gradients/Identity_14*-
_class#
!loc:@batch_normalization_2/beta*
valueB
 *    *
dtype0*
_output_shapes
: 
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
N*%
_output_shapes
:���������: *
T0*-
_class#
!loc:@batch_normalization_2/beta
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
Htraining/Adam/gradients/batch_normalization_2/moments/Squeeze_grad/ShapeConst*
dtype0*
_output_shapes
:*8
_class.
,*loc:@batch_normalization_2/moments/Squeeze*
valueB"   �   
�
Jtraining/Adam/gradients/batch_normalization_2/moments/Squeeze_grad/ReshapeReshapeFtraining/Adam/gradients/batch_normalization_2/batchnorm/mul_2_grad/MulHtraining/Adam/gradients/batch_normalization_2/moments/Squeeze_grad/Shape*
T0*8
_class.
,*loc:@batch_normalization_2/moments/Squeeze*
Tshape0*
_output_shapes
:	�
�
training/Adam/gradients/AddN_14AddNLtraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/Reshape_1Htraining/Adam/gradients/batch_normalization_2/batchnorm/mul_2_grad/Mul_1*
N*
_output_shapes	
:�*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/mul_1
�
Dtraining/Adam/gradients/batch_normalization_2/batchnorm/mul_grad/MulMultraining/Adam/gradients/AddN_142batch_normalization_2/batchnorm/mul/ReadVariableOp*
T0*6
_class,
*(loc:@batch_normalization_2/batchnorm/mul*
_output_shapes	
:�
�
Ftraining/Adam/gradients/batch_normalization_2/batchnorm/mul_grad/Mul_1Multraining/Adam/gradients/AddN_14%batch_normalization_2/batchnorm/Rsqrt*
_output_shapes	
:�*
T0*6
_class,
*(loc:@batch_normalization_2/batchnorm/mul
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
 training/Adam/gradients/zeros_15Fill'training/Adam/gradients/VariableShape_5&training/Adam/gradients/zeros_15/Const*#
_output_shapes
:���������*
T0*.
_class$
" loc:@batch_normalization_2/gamma*

index_type0
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
Vtraining/Adam/gradients/batch_normalization_2/batchnorm/add_grad/Sum/reduction_indicesConst*
dtype0*
_output_shapes
:*6
_class,
*(loc:@batch_normalization_2/batchnorm/add*
valueB: 
�
Dtraining/Adam/gradients/batch_normalization_2/batchnorm/add_grad/SumSumLtraining/Adam/gradients/batch_normalization_2/batchnorm/Rsqrt_grad/RsqrtGradVtraining/Adam/gradients/batch_normalization_2/batchnorm/add_grad/Sum/reduction_indices*
T0*6
_class,
*(loc:@batch_normalization_2/batchnorm/add*
_output_shapes
: *
	keep_dims( *

Tidx0
�
Ntraining/Adam/gradients/batch_normalization_2/batchnorm/add_grad/Reshape/shapeConst*6
_class,
*(loc:@batch_normalization_2/batchnorm/add*
valueB *
dtype0*
_output_shapes
: 
�
Htraining/Adam/gradients/batch_normalization_2/batchnorm/add_grad/ReshapeReshapeDtraining/Adam/gradients/batch_normalization_2/batchnorm/add_grad/SumNtraining/Adam/gradients/batch_normalization_2/batchnorm/add_grad/Reshape/shape*
T0*6
_class,
*(loc:@batch_normalization_2/batchnorm/add*
Tshape0*
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
Jtraining/Adam/gradients/batch_normalization_2/moments/Squeeze_1_grad/ShapeConst*:
_class0
.,loc:@batch_normalization_2/moments/Squeeze_1*
valueB"   �   *
dtype0*
_output_shapes
:
�
Ltraining/Adam/gradients/batch_normalization_2/moments/Squeeze_1_grad/ReshapeReshapeLtraining/Adam/gradients/batch_normalization_2/batchnorm/Rsqrt_grad/RsqrtGradJtraining/Adam/gradients/batch_normalization_2/moments/Squeeze_1_grad/Shape*
T0*:
_class0
.,loc:@batch_normalization_2/moments/Squeeze_1*
Tshape0*
_output_shapes
:	�
�
Itraining/Adam/gradients/batch_normalization_2/moments/variance_grad/ShapeShape/batch_normalization_2/moments/SquaredDifference*
T0*9
_class/
-+loc:@batch_normalization_2/moments/variance*
out_type0*
_output_shapes
:
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
Itraining/Adam/gradients/batch_normalization_2/moments/variance_grad/rangeRangeOtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/range/startHtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/SizeOtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/range/delta*9
_class/
-+loc:@batch_normalization_2/moments/variance*
_output_shapes
:*

Tidx0
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
T0*9
_class/
-+loc:@batch_normalization_2/moments/variance*
N*
_output_shapes
:
�
Mtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/Maximum/yConst*9
_class/
-+loc:@batch_normalization_2/moments/variance*
value	B :*
dtype0*
_output_shapes
: 
�
Ktraining/Adam/gradients/batch_normalization_2/moments/variance_grad/MaximumMaximumQtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/DynamicStitchMtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/Maximum/y*
T0*9
_class/
-+loc:@batch_normalization_2/moments/variance*
_output_shapes
:
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
Htraining/Adam/gradients/batch_normalization_2/moments/variance_grad/TileTileKtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/ReshapeLtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/floordiv*0
_output_shapes
:������������������*

Tmultiples0*
T0*9
_class/
-+loc:@batch_normalization_2/moments/variance
�
Ktraining/Adam/gradients/batch_normalization_2/moments/variance_grad/Shape_2Shape/batch_normalization_2/moments/SquaredDifference*
T0*9
_class/
-+loc:@batch_normalization_2/moments/variance*
out_type0*
_output_shapes
:
�
Ktraining/Adam/gradients/batch_normalization_2/moments/variance_grad/Shape_3Const*
dtype0*
_output_shapes
:*9
_class/
-+loc:@batch_normalization_2/moments/variance*
valueB"   �   
�
Itraining/Adam/gradients/batch_normalization_2/moments/variance_grad/ConstConst*
dtype0*
_output_shapes
:*9
_class/
-+loc:@batch_normalization_2/moments/variance*
valueB: 
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
Ktraining/Adam/gradients/batch_normalization_2/moments/variance_grad/Const_1Const*9
_class/
-+loc:@batch_normalization_2/moments/variance*
valueB: *
dtype0*
_output_shapes
:
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
Ktraining/Adam/gradients/batch_normalization_2/moments/variance_grad/truedivRealDivHtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/TileHtraining/Adam/gradients/batch_normalization_2/moments/variance_grad/Cast*
T0*9
_class/
-+loc:@batch_normalization_2/moments/variance*(
_output_shapes
:����������
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
Rtraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/mul_1MulPtraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/MulPtraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/sub*(
_output_shapes
:����������*
T0*B
_class8
64loc:@batch_normalization_2/moments/SquaredDifference
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
Rtraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/Sum_1SumRtraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/mul_1dtraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*B
_class8
64loc:@batch_normalization_2/moments/SquaredDifference
�
Vtraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/Reshape_1ReshapeRtraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/Sum_1Ttraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/Shape_1*
T0*B
_class8
64loc:@batch_normalization_2/moments/SquaredDifference*
Tshape0*
_output_shapes
:	�
�
Ptraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/NegNegVtraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/Reshape_1*
_output_shapes
:	�*
T0*B
_class8
64loc:@batch_normalization_2/moments/SquaredDifference
�
Etraining/Adam/gradients/batch_normalization_2/moments/mean_grad/ShapeShapedense_2/Relu*
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean*
out_type0*
_output_shapes
:
�
Dtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/SizeConst*
dtype0*
_output_shapes
: *5
_class+
)'loc:@batch_normalization_2/moments/mean*
value	B :
�
Ctraining/Adam/gradients/batch_normalization_2/moments/mean_grad/addAddV24batch_normalization_2/moments/mean/reduction_indicesDtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Size*
_output_shapes
:*
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean
�
Ctraining/Adam/gradients/batch_normalization_2/moments/mean_grad/modFloorModCtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/addDtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Size*
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean*
_output_shapes
:
�
Gtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Shape_1Const*5
_class+
)'loc:@batch_normalization_2/moments/mean*
valueB:*
dtype0*
_output_shapes
:
�
Ktraining/Adam/gradients/batch_normalization_2/moments/mean_grad/range/startConst*5
_class+
)'loc:@batch_normalization_2/moments/mean*
value	B : *
dtype0*
_output_shapes
: 
�
Ktraining/Adam/gradients/batch_normalization_2/moments/mean_grad/range/deltaConst*5
_class+
)'loc:@batch_normalization_2/moments/mean*
value	B :*
dtype0*
_output_shapes
: 
�
Etraining/Adam/gradients/batch_normalization_2/moments/mean_grad/rangeRangeKtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/range/startDtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/SizeKtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/range/delta*
_output_shapes
:*

Tidx0*5
_class+
)'loc:@batch_normalization_2/moments/mean
�
Jtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Fill/valueConst*
dtype0*
_output_shapes
: *5
_class+
)'loc:@batch_normalization_2/moments/mean*
value	B :
�
Dtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/FillFillGtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Shape_1Jtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Fill/value*
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean*

index_type0*
_output_shapes
:
�
Mtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/DynamicStitchDynamicStitchEtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/rangeCtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/modEtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/ShapeDtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Fill*
N*
_output_shapes
:*
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean
�
Itraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Maximum/yConst*
dtype0*
_output_shapes
: *5
_class+
)'loc:@batch_normalization_2/moments/mean*
value	B :
�
Gtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/MaximumMaximumMtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/DynamicStitchItraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Maximum/y*
_output_shapes
:*
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean
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
Gtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Shape_3Const*
dtype0*
_output_shapes
:*5
_class+
)'loc:@batch_normalization_2/moments/mean*
valueB"   �   
�
Etraining/Adam/gradients/batch_normalization_2/moments/mean_grad/ConstConst*5
_class+
)'loc:@batch_normalization_2/moments/mean*
valueB: *
dtype0*
_output_shapes
:
�
Dtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/ProdProdGtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Shape_2Etraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean
�
Gtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Const_1Const*5
_class+
)'loc:@batch_normalization_2/moments/mean*
valueB: *
dtype0*
_output_shapes
:
�
Ftraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Prod_1ProdGtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Shape_3Gtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Const_1*
	keep_dims( *

Tidx0*
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean*
_output_shapes
: 
�
Ktraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Maximum_1/yConst*5
_class+
)'loc:@batch_normalization_2/moments/mean*
value	B :*
dtype0*
_output_shapes
: 
�
Itraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Maximum_1MaximumFtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Prod_1Ktraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Maximum_1/y*
_output_shapes
: *
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean
�
Jtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/floordiv_1FloorDivDtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/ProdItraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Maximum_1*
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean*
_output_shapes
: 
�
Dtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/CastCastJtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/floordiv_1*
Truncate( *

DstT0*
_output_shapes
: *

SrcT0*5
_class+
)'loc:@batch_normalization_2/moments/mean
�
Gtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/truedivRealDivDtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/TileDtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/Cast*
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean*(
_output_shapes
:����������
�
training/Adam/gradients/AddN_16AddNXtraining/Adam/gradients/batch_normalization_2/cond/batchnorm/mul_1/Switch_grad/cond_gradJtraining/Adam/gradients/batch_normalization_2/batchnorm/mul_1_grad/ReshapeTtraining/Adam/gradients/batch_normalization_2/moments/SquaredDifference_grad/ReshapeGtraining/Adam/gradients/batch_normalization_2/moments/mean_grad/truediv*
T0*
_class
loc:@dense_2/Relu*
N*(
_output_shapes
:����������
�
2training/Adam/gradients/dense_2/Relu_grad/ReluGradReluGradtraining/Adam/gradients/AddN_16dense_2/Relu*
T0*
_class
loc:@dense_2/Relu*(
_output_shapes
:����������
�
8training/Adam/gradients/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad2training/Adam/gradients/dense_2/Relu_grad/ReluGrad*
T0*"
_class
loc:@dense_2/BiasAdd*
data_formatNHWC*
_output_shapes	
:�
�
2training/Adam/gradients/dense_2/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_2/Relu_grad/ReluGraddense_2/MatMul/ReadVariableOp*
transpose_a( *(
_output_shapes
:����������*
transpose_b(*
T0*!
_class
loc:@dense_2/MatMul
�
4training/Adam/gradients/dense_2/MatMul_grad/MatMul_1MatMul batch_normalization_1/cond/Merge2training/Adam/gradients/dense_2/Relu_grad/ReluGrad*
transpose_a(* 
_output_shapes
:
��*
transpose_b( *
T0*!
_class
loc:@dense_2/MatMul
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
Otraining/Adam/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/Shape_1Shape(batch_normalization_1/cond/batchnorm/sub*
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/add_1*
out_type0*
_output_shapes
:
�
]training/Adam/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsMtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/ShapeOtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/add_1
�
Ktraining/Adam/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/SumSumGtraining/Adam/gradients/batch_normalization_1/cond/Merge_grad/cond_grad]training/Adam/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/add_1
�
Otraining/Adam/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/ReshapeReshapeKtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/SumMtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/Shape*
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/add_1*
Tshape0*(
_output_shapes
:����������
�
Mtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/Sum_1SumGtraining/Adam/gradients/batch_normalization_1/cond/Merge_grad/cond_grad_training/Adam/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/add_1
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
#training/Adam/gradients/Identity_16Identity!training/Adam/gradients/Switch_16*(
_output_shapes
:����������*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1
�
 training/Adam/gradients/Shape_11Shape!training/Adam/gradients/Switch_16*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1*
out_type0*
_output_shapes
:
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
 training/Adam/gradients/zeros_16Fill training/Adam/gradients/Shape_11&training/Adam/gradients/zeros_16/Const*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1*

index_type0*(
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
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/mul_1*
out_type0*
_output_shapes
:
�
Otraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/Shape_1Shape(batch_normalization_1/cond/batchnorm/mul*
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/mul_1*
out_type0*
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
Ktraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/SumSumKtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/Mul]training/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/BroadcastGradientArgs*
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/mul_1*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Otraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/ReshapeReshapeKtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/SumMtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/Shape*
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/mul_1*
Tshape0*(
_output_shapes
:����������
�
Mtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/Mul_1Mul1batch_normalization_1/cond/batchnorm/mul_1/SwitchOtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/Reshape*
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/mul_1*(
_output_shapes
:����������
�
Mtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/Sum_1SumMtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/Mul_1_training/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/mul_1
�
Qtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/Reshape_1ReshapeMtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/Sum_1Otraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/Shape_1*
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/mul_1*
Tshape0*
_output_shapes	
:�
�
Itraining/Adam/gradients/batch_normalization_1/cond/batchnorm/sub_grad/NegNegQtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/Reshape_1*
T0*;
_class1
/-loc:@batch_normalization_1/cond/batchnorm/sub*
_output_shapes	
:�
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
Ftraining/Adam/gradients/batch_normalization_1/batchnorm/add_1_grad/SumSumJtraining/Adam/gradients/batch_normalization_1/cond/Switch_1_grad/cond_gradXtraining/Adam/gradients/batch_normalization_1/batchnorm/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1
�
Jtraining/Adam/gradients/batch_normalization_1/batchnorm/add_1_grad/ReshapeReshapeFtraining/Adam/gradients/batch_normalization_1/batchnorm/add_1_grad/SumHtraining/Adam/gradients/batch_normalization_1/batchnorm/add_1_grad/Shape*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1*
Tshape0*(
_output_shapes
:����������
�
Htraining/Adam/gradients/batch_normalization_1/batchnorm/add_1_grad/Sum_1SumJtraining/Adam/gradients/batch_normalization_1/cond/Switch_1_grad/cond_gradZtraining/Adam/gradients/batch_normalization_1/batchnorm/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1
�
Ltraining/Adam/gradients/batch_normalization_1/batchnorm/add_1_grad/Reshape_1ReshapeHtraining/Adam/gradients/batch_normalization_1/batchnorm/add_1_grad/Sum_1Jtraining/Adam/gradients/batch_normalization_1/batchnorm/add_1_grad/Shape_1*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1*
Tshape0*
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
#training/Adam/gradients/Identity_17Identity#training/Adam/gradients/Switch_17:1*
T0*
_class
loc:@dense_1/Relu*(
_output_shapes
:����������
�
 training/Adam/gradients/Shape_12Shape#training/Adam/gradients/Switch_17:1*
T0*
_class
loc:@dense_1/Relu*
out_type0*
_output_shapes
:
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
 training/Adam/gradients/zeros_17Fill training/Adam/gradients/Shape_12&training/Adam/gradients/zeros_17/Const*(
_output_shapes
:����������*
T0*
_class
loc:@dense_1/Relu*

index_type0
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
Mtraining/Adam/gradients/batch_normalization_1/cond/batchnorm/mul_2_grad/Mul_1MulItraining/Adam/gradients/batch_normalization_1/cond/batchnorm/sub_grad/Neg5batch_normalization_1/cond/batchnorm/ReadVariableOp_1*
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/mul_2*
_output_shapes	
:�
�
Htraining/Adam/gradients/batch_normalization_1/batchnorm/mul_1_grad/ShapeShapedense_1/Relu*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/mul_1*
out_type0*
_output_shapes
:
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
 training/Adam/gradients/zeros_18Fill'training/Adam/gradients/VariableShape_6&training/Adam/gradients/zeros_18/Const*#
_output_shapes
:���������*
T0*-
_class#
!loc:@batch_normalization_1/beta*

index_type0
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
T0*-
_class#
!loc:@batch_normalization_1/beta*
N*
_output_shapes	
:�
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
Dtraining/Adam/gradients/batch_normalization_1/batchnorm/mul_grad/MulMultraining/Adam/gradients/AddN_192batch_normalization_1/batchnorm/mul/ReadVariableOp*
_output_shapes	
:�*
T0*6
_class,
*(loc:@batch_normalization_1/batchnorm/mul
�
Ftraining/Adam/gradients/batch_normalization_1/batchnorm/mul_grad/Mul_1Multraining/Adam/gradients/AddN_19%batch_normalization_1/batchnorm/Rsqrt*
_output_shapes	
:�*
T0*6
_class,
*(loc:@batch_normalization_1/batchnorm/mul
�
!training/Adam/gradients/Switch_19Switchbatch_normalization_1/gamma"batch_normalization_1/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_1/gamma*
_output_shapes
: : 
�
#training/Adam/gradients/Identity_19Identity#training/Adam/gradients/Switch_19:1*
_output_shapes
: *
T0*.
_class$
" loc:@batch_normalization_1/gamma
�
'training/Adam/gradients/VariableShape_7VariableShape#training/Adam/gradients/Switch_19:1$^training/Adam/gradients/Identity_19*.
_class$
" loc:@batch_normalization_1/gamma*
out_type0*
_output_shapes
:
�
&training/Adam/gradients/zeros_19/ConstConst$^training/Adam/gradients/Identity_19*.
_class$
" loc:@batch_normalization_1/gamma*
valueB
 *    *
dtype0*
_output_shapes
: 
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
Ltraining/Adam/gradients/batch_normalization_1/batchnorm/Rsqrt_grad/RsqrtGrad	RsqrtGrad%batch_normalization_1/batchnorm/RsqrtDtraining/Adam/gradients/batch_normalization_1/batchnorm/mul_grad/Mul*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/Rsqrt*
_output_shapes	
:�
�
Vtraining/Adam/gradients/batch_normalization_1/batchnorm/add_grad/Sum/reduction_indicesConst*
dtype0*
_output_shapes
:*6
_class,
*(loc:@batch_normalization_1/batchnorm/add*
valueB: 
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
Ntraining/Adam/gradients/batch_normalization_1/batchnorm/add_grad/Reshape/shapeConst*6
_class,
*(loc:@batch_normalization_1/batchnorm/add*
valueB *
dtype0*
_output_shapes
: 
�
Htraining/Adam/gradients/batch_normalization_1/batchnorm/add_grad/ReshapeReshapeDtraining/Adam/gradients/batch_normalization_1/batchnorm/add_grad/SumNtraining/Adam/gradients/batch_normalization_1/batchnorm/add_grad/Reshape/shape*
_output_shapes
: *
T0*6
_class,
*(loc:@batch_normalization_1/batchnorm/add*
Tshape0
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
Ltraining/Adam/gradients/batch_normalization_1/moments/Squeeze_1_grad/ReshapeReshapeLtraining/Adam/gradients/batch_normalization_1/batchnorm/Rsqrt_grad/RsqrtGradJtraining/Adam/gradients/batch_normalization_1/moments/Squeeze_1_grad/Shape*
_output_shapes
:	�*
T0*:
_class0
.,loc:@batch_normalization_1/moments/Squeeze_1*
Tshape0
�
Itraining/Adam/gradients/batch_normalization_1/moments/variance_grad/ShapeShape/batch_normalization_1/moments/SquaredDifference*
T0*9
_class/
-+loc:@batch_normalization_1/moments/variance*
out_type0*
_output_shapes
:
�
Htraining/Adam/gradients/batch_normalization_1/moments/variance_grad/SizeConst*
dtype0*
_output_shapes
: *9
_class/
-+loc:@batch_normalization_1/moments/variance*
value	B :
�
Gtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/addAddV28batch_normalization_1/moments/variance/reduction_indicesHtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Size*
T0*9
_class/
-+loc:@batch_normalization_1/moments/variance*
_output_shapes
:
�
Gtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/modFloorModGtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/addHtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Size*
_output_shapes
:*
T0*9
_class/
-+loc:@batch_normalization_1/moments/variance
�
Ktraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Shape_1Const*9
_class/
-+loc:@batch_normalization_1/moments/variance*
valueB:*
dtype0*
_output_shapes
:
�
Otraining/Adam/gradients/batch_normalization_1/moments/variance_grad/range/startConst*9
_class/
-+loc:@batch_normalization_1/moments/variance*
value	B : *
dtype0*
_output_shapes
: 
�
Otraining/Adam/gradients/batch_normalization_1/moments/variance_grad/range/deltaConst*
dtype0*
_output_shapes
: *9
_class/
-+loc:@batch_normalization_1/moments/variance*
value	B :
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
N*
_output_shapes
:*
T0*9
_class/
-+loc:@batch_normalization_1/moments/variance
�
Mtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Maximum/yConst*9
_class/
-+loc:@batch_normalization_1/moments/variance*
value	B :*
dtype0*
_output_shapes
: 
�
Ktraining/Adam/gradients/batch_normalization_1/moments/variance_grad/MaximumMaximumQtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/DynamicStitchMtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Maximum/y*
_output_shapes
:*
T0*9
_class/
-+loc:@batch_normalization_1/moments/variance
�
Ltraining/Adam/gradients/batch_normalization_1/moments/variance_grad/floordivFloorDivItraining/Adam/gradients/batch_normalization_1/moments/variance_grad/ShapeKtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Maximum*
_output_shapes
:*
T0*9
_class/
-+loc:@batch_normalization_1/moments/variance
�
Ktraining/Adam/gradients/batch_normalization_1/moments/variance_grad/ReshapeReshapeLtraining/Adam/gradients/batch_normalization_1/moments/Squeeze_1_grad/ReshapeQtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/DynamicStitch*
T0*9
_class/
-+loc:@batch_normalization_1/moments/variance*
Tshape0*0
_output_shapes
:������������������
�
Htraining/Adam/gradients/batch_normalization_1/moments/variance_grad/TileTileKtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/ReshapeLtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/floordiv*

Tmultiples0*
T0*9
_class/
-+loc:@batch_normalization_1/moments/variance*0
_output_shapes
:������������������
�
Ktraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Shape_2Shape/batch_normalization_1/moments/SquaredDifference*
_output_shapes
:*
T0*9
_class/
-+loc:@batch_normalization_1/moments/variance*
out_type0
�
Ktraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Shape_3Const*9
_class/
-+loc:@batch_normalization_1/moments/variance*
valueB"   �   *
dtype0*
_output_shapes
:
�
Itraining/Adam/gradients/batch_normalization_1/moments/variance_grad/ConstConst*9
_class/
-+loc:@batch_normalization_1/moments/variance*
valueB: *
dtype0*
_output_shapes
:
�
Htraining/Adam/gradients/batch_normalization_1/moments/variance_grad/ProdProdKtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Shape_2Itraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Const*
T0*9
_class/
-+loc:@batch_normalization_1/moments/variance*
_output_shapes
: *
	keep_dims( *

Tidx0
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
Otraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Maximum_1/yConst*9
_class/
-+loc:@batch_normalization_1/moments/variance*
value	B :*
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
Ntraining/Adam/gradients/batch_normalization_1/moments/variance_grad/floordiv_1FloorDivHtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/ProdMtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/Maximum_1*
_output_shapes
: *
T0*9
_class/
-+loc:@batch_normalization_1/moments/variance
�
Htraining/Adam/gradients/batch_normalization_1/moments/variance_grad/CastCastNtraining/Adam/gradients/batch_normalization_1/moments/variance_grad/floordiv_1*
Truncate( *

DstT0*
_output_shapes
: *

SrcT0*9
_class/
-+loc:@batch_normalization_1/moments/variance
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
Ptraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/subSubdense_1/Relu*batch_normalization_1/moments/StopGradientL^training/Adam/gradients/batch_normalization_1/moments/variance_grad/truediv*(
_output_shapes
:����������*
T0*B
_class8
64loc:@batch_normalization_1/moments/SquaredDifference
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
Ttraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/Shape_1Shape*batch_normalization_1/moments/StopGradient*
T0*B
_class8
64loc:@batch_normalization_1/moments/SquaredDifference*
out_type0*
_output_shapes
:
�
btraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgsRtraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/ShapeTtraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/Shape_1*
T0*B
_class8
64loc:@batch_normalization_1/moments/SquaredDifference*2
_output_shapes 
:���������:���������
�
Ptraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/SumSumRtraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/mul_1btraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*B
_class8
64loc:@batch_normalization_1/moments/SquaredDifference
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
Ptraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/NegNegVtraining/Adam/gradients/batch_normalization_1/moments/SquaredDifference_grad/Reshape_1*
_output_shapes
:	�*
T0*B
_class8
64loc:@batch_normalization_1/moments/SquaredDifference
�
Etraining/Adam/gradients/batch_normalization_1/moments/mean_grad/ShapeShapedense_1/Relu*
_output_shapes
:*
T0*5
_class+
)'loc:@batch_normalization_1/moments/mean*
out_type0
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
Ctraining/Adam/gradients/batch_normalization_1/moments/mean_grad/modFloorModCtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/addDtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Size*
_output_shapes
:*
T0*5
_class+
)'loc:@batch_normalization_1/moments/mean
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
Ktraining/Adam/gradients/batch_normalization_1/moments/mean_grad/range/deltaConst*5
_class+
)'loc:@batch_normalization_1/moments/mean*
value	B :*
dtype0*
_output_shapes
: 
�
Etraining/Adam/gradients/batch_normalization_1/moments/mean_grad/rangeRangeKtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/range/startDtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/SizeKtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/range/delta*5
_class+
)'loc:@batch_normalization_1/moments/mean*
_output_shapes
:*

Tidx0
�
Jtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Fill/valueConst*5
_class+
)'loc:@batch_normalization_1/moments/mean*
value	B :*
dtype0*
_output_shapes
: 
�
Dtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/FillFillGtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Shape_1Jtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Fill/value*
T0*5
_class+
)'loc:@batch_normalization_1/moments/mean*

index_type0*
_output_shapes
:
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
Gtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/MaximumMaximumMtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/DynamicStitchItraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Maximum/y*
_output_shapes
:*
T0*5
_class+
)'loc:@batch_normalization_1/moments/mean
�
Htraining/Adam/gradients/batch_normalization_1/moments/mean_grad/floordivFloorDivEtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/ShapeGtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Maximum*
_output_shapes
:*
T0*5
_class+
)'loc:@batch_normalization_1/moments/mean
�
Gtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/ReshapeReshapeJtraining/Adam/gradients/batch_normalization_1/moments/Squeeze_grad/ReshapeMtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/DynamicStitch*
T0*5
_class+
)'loc:@batch_normalization_1/moments/mean*
Tshape0*0
_output_shapes
:������������������
�
Dtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/TileTileGtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/ReshapeHtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/floordiv*
T0*5
_class+
)'loc:@batch_normalization_1/moments/mean*0
_output_shapes
:������������������*

Tmultiples0
�
Gtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Shape_2Shapedense_1/Relu*
T0*5
_class+
)'loc:@batch_normalization_1/moments/mean*
out_type0*
_output_shapes
:
�
Gtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Shape_3Const*
dtype0*
_output_shapes
:*5
_class+
)'loc:@batch_normalization_1/moments/mean*
valueB"   �   
�
Etraining/Adam/gradients/batch_normalization_1/moments/mean_grad/ConstConst*5
_class+
)'loc:@batch_normalization_1/moments/mean*
valueB: *
dtype0*
_output_shapes
:
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
Ftraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Prod_1ProdGtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Shape_3Gtraining/Adam/gradients/batch_normalization_1/moments/mean_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0*5
_class+
)'loc:@batch_normalization_1/moments/mean
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
2training/Adam/gradients/dense_1/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_1/Relu_grad/ReluGraddense_1/MatMul/ReadVariableOp*
T0*!
_class
loc:@dense_1/MatMul*
transpose_a( *(
_output_shapes
:����������*
transpose_b(
�
4training/Adam/gradients/dense_1/MatMul_grad/MatMul_1MatMuldense_1_input2training/Adam/gradients/dense_1/Relu_grad/ReluGrad*
transpose_b( *
T0*!
_class
loc:@dense_1/MatMul*
transpose_a(* 
_output_shapes
:
��
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
training/Adam/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
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
training/Adam/sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
a
training/Adam/subSubtraining/Adam/sub/xtraining/Adam/Pow*
_output_shapes
: *
T0
Z
training/Adam/Const_1Const*
valueB
 *    *
dtype0*
_output_shapes
: 
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
valueB"   �   *
dtype0*
_output_shapes
:
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
training/Adam/m_2_1VarHandleOp*
dtype0*
_output_shapes
: *$
shared_nametraining/Adam/m_2_1*&
_class
loc:@training/Adam/m_2_1*
	container *
shape:�
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
training/Adam/m_3_1VarHandleOp*&
_class
loc:@training/Adam/m_3_1*
	container *
shape:�*
dtype0*
_output_shapes
: *$
shared_nametraining/Adam/m_3_1
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
dtype0*
_output_shapes
:*
valueB"�   �   
\
training/Adam/m_4/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
�
training/Adam/m_4Fill!training/Adam/m_4/shape_as_tensortraining/Adam/m_4/Const*
T0*

index_type0* 
_output_shapes
:
��
�
training/Adam/m_4_1VarHandleOp*
	container *
shape:
��*
dtype0*
_output_shapes
: *$
shared_nametraining/Adam/m_4_1*&
_class
loc:@training/Adam/m_4_1
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
training/Adam/m_5_1VarHandleOp*$
shared_nametraining/Adam/m_5_1*&
_class
loc:@training/Adam/m_5_1*
	container *
shape:�*
dtype0*
_output_shapes
: 
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
training/Adam/m_6_1VarHandleOp*&
_class
loc:@training/Adam/m_6_1*
	container *
shape:�*
dtype0*
_output_shapes
: *$
shared_nametraining/Adam/m_6_1
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
training/Adam/m_7_1VarHandleOp*&
_class
loc:@training/Adam/m_7_1*
	container *
shape:�*
dtype0*
_output_shapes
: *$
shared_nametraining/Adam/m_7_1
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
training/Adam/m_9_1VarHandleOp*
dtype0*
_output_shapes
: *$
shared_nametraining/Adam/m_9_1*&
_class
loc:@training/Adam/m_9_1*
	container *
shape:�
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
training/Adam/m_10Const*
dtype0*
_output_shapes	
:�*
valueB�*    
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
training/Adam/m_11_1VarHandleOp*%
shared_nametraining/Adam/m_11_1*'
_class
loc:@training/Adam/m_11_1*
	container *
shape:�*
dtype0*
_output_shapes
: 
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
training/Adam/m_12/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/m_12Fill"training/Adam/m_12/shape_as_tensortraining/Adam/m_12/Const*
T0*

index_type0* 
_output_shapes
:
��
�
training/Adam/m_12_1VarHandleOp*
shape:
��*
dtype0*
_output_shapes
: *%
shared_nametraining/Adam/m_12_1*'
_class
loc:@training/Adam/m_12_1*
	container 
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
training/Adam/m_14_1VarHandleOp*'
_class
loc:@training/Adam/m_14_1*
	container *
shape:�*
dtype0*
_output_shapes
: *%
shared_nametraining/Adam/m_14_1
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
dtype0*
_output_shapes
:*
valueB"�   +   
]
training/Adam/m_16/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
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
training/Adam/v_0_1VarHandleOp*$
shared_nametraining/Adam/v_0_1*&
_class
loc:@training/Adam/v_0_1*
	container *
shape:
��*
dtype0*
_output_shapes
: 
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
training/Adam/v_1_1VarHandleOp*
	container *
shape:�*
dtype0*
_output_shapes
: *$
shared_nametraining/Adam/v_1_1*&
_class
loc:@training/Adam/v_1_1
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
training/Adam/v_2_1VarHandleOp*
	container *
shape:�*
dtype0*
_output_shapes
: *$
shared_nametraining/Adam/v_2_1*&
_class
loc:@training/Adam/v_2_1
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
training/Adam/v_3_1VarHandleOp*
dtype0*
_output_shapes
: *$
shared_nametraining/Adam/v_3_1*&
_class
loc:@training/Adam/v_3_1*
	container *
shape:�
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
training/Adam/v_4_1VarHandleOp*
dtype0*
_output_shapes
: *$
shared_nametraining/Adam/v_4_1*&
_class
loc:@training/Adam/v_4_1*
	container *
shape:
��
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
training/Adam/v_5_1VarHandleOp*&
_class
loc:@training/Adam/v_5_1*
	container *
shape:�*
dtype0*
_output_shapes
: *$
shared_nametraining/Adam/v_5_1
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
training/Adam/v_6_1VarHandleOp*
	container *
shape:�*
dtype0*
_output_shapes
: *$
shared_nametraining/Adam/v_6_1*&
_class
loc:@training/Adam/v_6_1
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
training/Adam/v_7_1VarHandleOp*
dtype0*
_output_shapes
: *$
shared_nametraining/Adam/v_7_1*&
_class
loc:@training/Adam/v_7_1*
	container *
shape:�
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
training/Adam/v_8_1VarHandleOp*$
shared_nametraining/Adam/v_8_1*&
_class
loc:@training/Adam/v_8_1*
	container *
shape:
��*
dtype0*
_output_shapes
: 
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
training/Adam/v_9_1VarHandleOp*$
shared_nametraining/Adam/v_9_1*&
_class
loc:@training/Adam/v_9_1*
	container *
shape:�*
dtype0*
_output_shapes
: 
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
training/Adam/v_10_1VarHandleOp*'
_class
loc:@training/Adam/v_10_1*
	container *
shape:�*
dtype0*
_output_shapes
: *%
shared_nametraining/Adam/v_10_1
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
training/Adam/v_14_1VarHandleOp*
shape:�*
dtype0*
_output_shapes
: *%
shared_nametraining/Adam/v_14_1*'
_class
loc:@training/Adam/v_14_1*
	container 
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
training/Adam/v_15_1VarHandleOp*%
shared_nametraining/Adam/v_15_1*'
_class
loc:@training/Adam/v_15_1*
	container *
shape:�*
dtype0*
_output_shapes
: 
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
training/Adam/v_16/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/v_16Fill"training/Adam/v_16/shape_as_tensortraining/Adam/v_16/Const*
T0*

index_type0*
_output_shapes
:	�+
�
training/Adam/v_16_1VarHandleOp*%
shared_nametraining/Adam/v_16_1*'
_class
loc:@training/Adam/v_16_1*
	container *
shape:	�+*
dtype0*
_output_shapes
: 
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
training/Adam/vhat_1Fill$training/Adam/vhat_1/shape_as_tensortraining/Adam/vhat_1/Const*
_output_shapes
:*
T0*

index_type0
�
training/Adam/vhat_1_1VarHandleOp*
dtype0*
_output_shapes
: *'
shared_nametraining/Adam/vhat_1_1*)
_class
loc:@training/Adam/vhat_1_1*
	container *
shape:
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
training/Adam/vhat_2Fill$training/Adam/vhat_2/shape_as_tensortraining/Adam/vhat_2/Const*
T0*

index_type0*
_output_shapes
:
�
training/Adam/vhat_2_1VarHandleOp*
	container *
shape:*
dtype0*
_output_shapes
: *'
shared_nametraining/Adam/vhat_2_1*)
_class
loc:@training/Adam/vhat_2_1
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
$training/Adam/vhat_4/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
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
training/Adam/vhat_4_1VarHandleOp*
	container *
shape:*
dtype0*
_output_shapes
: *'
shared_nametraining/Adam/vhat_4_1*)
_class
loc:@training/Adam/vhat_4_1
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
$training/Adam/vhat_5/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
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
training/Adam/vhat_5_1VarHandleOp*
dtype0*
_output_shapes
: *'
shared_nametraining/Adam/vhat_5_1*)
_class
loc:@training/Adam/vhat_5_1*
	container *
shape:
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
training/Adam/vhat_6/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
�
training/Adam/vhat_6Fill$training/Adam/vhat_6/shape_as_tensortraining/Adam/vhat_6/Const*
_output_shapes
:*
T0*

index_type0
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
training/Adam/vhat_7/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
�
training/Adam/vhat_7Fill$training/Adam/vhat_7/shape_as_tensortraining/Adam/vhat_7/Const*
T0*

index_type0*
_output_shapes
:
�
training/Adam/vhat_7_1VarHandleOp*
	container *
shape:*
dtype0*
_output_shapes
: *'
shared_nametraining/Adam/vhat_7_1*)
_class
loc:@training/Adam/vhat_7_1
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
$training/Adam/vhat_8/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
_
training/Adam/vhat_8/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/vhat_8Fill$training/Adam/vhat_8/shape_as_tensortraining/Adam/vhat_8/Const*
_output_shapes
:*
T0*

index_type0
�
training/Adam/vhat_8_1VarHandleOp*
	container *
shape:*
dtype0*
_output_shapes
: *'
shared_nametraining/Adam/vhat_8_1*)
_class
loc:@training/Adam/vhat_8_1
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
$training/Adam/vhat_9/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
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
training/Adam/vhat_10_1VarHandleOp**
_class 
loc:@training/Adam/vhat_10_1*
	container *
shape:*
dtype0*
_output_shapes
: *(
shared_nametraining/Adam/vhat_10_1
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
training/Adam/vhat_11Fill%training/Adam/vhat_11/shape_as_tensortraining/Adam/vhat_11/Const*
_output_shapes
:*
T0*

index_type0
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
%training/Adam/vhat_12/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
`
training/Adam/vhat_12/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/vhat_12Fill%training/Adam/vhat_12/shape_as_tensortraining/Adam/vhat_12/Const*
_output_shapes
:*
T0*

index_type0
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
training/Adam/vhat_13_1VarHandleOp*
dtype0*
_output_shapes
: *(
shared_nametraining/Adam/vhat_13_1**
_class 
loc:@training/Adam/vhat_13_1*
	container *
shape:
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
%training/Adam/vhat_14/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
`
training/Adam/vhat_14/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/vhat_14Fill%training/Adam/vhat_14/shape_as_tensortraining/Adam/vhat_14/Const*
_output_shapes
:*
T0*

index_type0
�
training/Adam/vhat_14_1VarHandleOp*(
shared_nametraining/Adam/vhat_14_1**
_class 
loc:@training/Adam/vhat_14_1*
	container *
shape:*
dtype0*
_output_shapes
: 
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
%training/Adam/vhat_16/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
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
training/Adam/vhat_16_1VarHandleOp*
dtype0*
_output_shapes
: *(
shared_nametraining/Adam/vhat_16_1**
_class 
loc:@training/Adam/vhat_16_1*
	container *
shape:
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
%training/Adam/vhat_17/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
`
training/Adam/vhat_17/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/vhat_17Fill%training/Adam/vhat_17/shape_as_tensortraining/Adam/vhat_17/Const*
T0*

index_type0*
_output_shapes
:
�
training/Adam/vhat_17_1VarHandleOp*
dtype0*
_output_shapes
: *(
shared_nametraining/Adam/vhat_17_1**
_class 
loc:@training/Adam/vhat_17_1*
	container *
shape:
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
training/Adam/mul_2Multraining/Adam/sub_24training/Adam/gradients/dense_1/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:
��
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
training/Adam/mul_3Multraining/Adam/ReadVariableOp_4"training/Adam/mul_3/ReadVariableOp*
T0* 
_output_shapes
:
��
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
training/Adam/sub_3Subtraining/Adam/sub_3/xtraining/Adam/ReadVariableOp_5*
T0*
_output_shapes
: 
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
training/Adam/add_2AddV2training/Adam/mul_3training/Adam/mul_4*
T0* 
_output_shapes
:
��
m
training/Adam/mul_5Multraining/Adam/multraining/Adam/add_1*
T0* 
_output_shapes
:
��
Z
training/Adam/Const_3Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_4Const*
dtype0*
_output_shapes
: *
valueB
 *  �
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
training/Adam/truediv_1RealDivtraining/Adam/mul_5training/Adam/add_3*
T0* 
_output_shapes
:
��
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
training/Adam/sub_6Subtraining/Adam/sub_6/xtraining/Adam/ReadVariableOp_13*
_output_shapes
: *
T0
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
training/Adam/add_5AddV2training/Adam/mul_8training/Adam/mul_9*
T0*
_output_shapes	
:�
i
training/Adam/mul_10Multraining/Adam/multraining/Adam/add_4*
_output_shapes	
:�*
T0
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
%training/Adam/clip_by_value_2/MinimumMinimumtraining/Adam/add_5training/Adam/Const_6*
_output_shapes	
:�*
T0
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
training/Adam/add_6/yConst*
dtype0*
_output_shapes
: *
valueB
 *���3
o
training/Adam/add_6AddV2training/Adam/Sqrt_2training/Adam/add_6/y*
_output_shapes	
:�*
T0
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
training/Adam/clip_by_value_3Maximum%training/Adam/clip_by_value_3/Minimumtraining/Adam/Const_7*
_output_shapes	
:�*
T0
a
training/Adam/Sqrt_3Sqrttraining/Adam/clip_by_value_3*
T0*
_output_shapes	
:�
Z
training/Adam/add_9/yConst*
dtype0*
_output_shapes
: *
valueB
 *���3
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
training/Adam/sub_11/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
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
training/Adam/mul_18Multraining/Adam/ReadVariableOp_28#training/Adam/mul_18/ReadVariableOp*
_output_shapes	
:�*
T0
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
training/Adam/add_11AddV2training/Adam/mul_18training/Adam/mul_19*
T0*
_output_shapes	
:�
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
training/Adam/Const_10Const*
dtype0*
_output_shapes
: *
valueB
 *  �
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
training/Adam/Sqrt_4Sqrttraining/Adam/clip_by_value_4*
_output_shapes	
:�*
T0
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
training/Adam/sub_15/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
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
training/Adam/mul_25Multraining/Adam/multraining/Adam/add_13* 
_output_shapes
:
��*
T0
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
training/Adam/add_15/yConst*
dtype0*
_output_shapes
: *
valueB
 *���3
v
training/Adam/add_15AddV2training/Adam/Sqrt_5training/Adam/add_15/y* 
_output_shapes
:
��*
T0
y
training/Adam/truediv_5RealDivtraining/Adam/mul_25training/Adam/add_15* 
_output_shapes
:
��*
T0
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
training/Adam/sub_18Subtraining/Adam/sub_18/xtraining/Adam/ReadVariableOp_45*
T0*
_output_shapes
: 
�
training/Adam/Square_5Square8training/Adam/gradients/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�*
T0
o
training/Adam/mul_29Multraining/Adam/sub_18training/Adam/Square_5*
T0*
_output_shapes	
:�
o
training/Adam/add_17AddV2training/Adam/mul_28training/Adam/mul_29*
_output_shapes	
:�*
T0
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
training/Adam/Const_14Const*
valueB
 *  �*
dtype0*
_output_shapes
: 
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
training/Adam/Sqrt_6Sqrttraining/Adam/clip_by_value_6*
_output_shapes	
:�*
T0
[
training/Adam/add_18/yConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
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
training/Adam/add_20AddV2training/Adam/mul_33training/Adam/mul_34*
_output_shapes	
:�*
T0
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
training/Adam/clip_by_value_7Maximum%training/Adam/clip_by_value_7/Minimumtraining/Adam/Const_15*
_output_shapes	
:�*
T0
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
training/Adam/mul_37Multraining/Adam/sub_23training/Adam/gradients/AddN_13*
T0*
_output_shapes	
:�
o
training/Adam/add_22AddV2training/Adam/mul_36training/Adam/mul_37*
_output_shapes	
:�*
T0
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
training/Adam/Const_18Const*
dtype0*
_output_shapes
: *
valueB
 *  �
�
%training/Adam/clip_by_value_8/MinimumMinimumtraining/Adam/add_23training/Adam/Const_18*
_output_shapes	
:�*
T0
�
training/Adam/clip_by_value_8Maximum%training/Adam/clip_by_value_8/Minimumtraining/Adam/Const_17*
T0*
_output_shapes	
:�
a
training/Adam/Sqrt_8Sqrttraining/Adam/clip_by_value_8*
T0*
_output_shapes	
:�
[
training/Adam/add_24/yConst*
dtype0*
_output_shapes
: *
valueB
 *���3
q
training/Adam/add_24AddV2training/Adam/Sqrt_8training/Adam/add_24/y*
_output_shapes	
:�*
T0
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
training/Adam/add_27/yConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
v
training/Adam/add_27AddV2training/Adam/Sqrt_9training/Adam/add_27/y*
T0* 
_output_shapes
:
��
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
training/Adam/sub_29/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
u
training/Adam/sub_29Subtraining/Adam/sub_29/xtraining/Adam/ReadVariableOp_75*
T0*
_output_shapes
: 
�
training/Adam/mul_47Multraining/Adam/sub_298training/Adam/gradients/dense_3/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:�
o
training/Adam/add_28AddV2training/Adam/mul_46training/Adam/mul_47*
_output_shapes	
:�*
T0
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
training/Adam/clip_by_value_10Maximum&training/Adam/clip_by_value_10/Minimumtraining/Adam/Const_21*
T0*
_output_shapes	
:�
c
training/Adam/Sqrt_10Sqrttraining/Adam/clip_by_value_10*
_output_shapes	
:�*
T0
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
training/Adam/add_31AddV2training/Adam/mul_51training/Adam/mul_52*
T0*
_output_shapes	
:�
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
training/Adam/clip_by_value_11Maximum&training/Adam/clip_by_value_11/Minimumtraining/Adam/Const_23*
_output_shapes	
:�*
T0
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
training/Adam/mul_58Multraining/Adam/ReadVariableOp_92#training/Adam/mul_58/ReadVariableOp*
_output_shapes	
:�*
T0
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
training/Adam/add_35AddV2training/Adam/mul_58training/Adam/mul_59*
_output_shapes	
:�*
T0
j
training/Adam/mul_60Multraining/Adam/multraining/Adam/add_34*
T0*
_output_shapes	
:�
[
training/Adam/Const_25Const*
dtype0*
_output_shapes
: *
valueB
 *    
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
training/Adam/truediv_12RealDivtraining/Adam/mul_60training/Adam/add_36*
_output_shapes	
:�*
T0
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
training/Adam/mul_61Multraining/Adam/ReadVariableOp_98#training/Adam/mul_61/ReadVariableOp*
T0* 
_output_shapes
:
��
c
training/Adam/ReadVariableOp_99ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
[
training/Adam/sub_38/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
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
training/Adam/sub_39/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
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
training/Adam/Const_28Const*
valueB
 *  �*
dtype0*
_output_shapes
: 
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
training/Adam/add_39/yConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
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
training/Adam/mul_67Multraining/Adam/sub_418training/Adam/gradients/dense_4/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:�
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
training/Adam/mul_69Multraining/Adam/sub_42training/Adam/Square_13*
T0*
_output_shapes	
:�
o
training/Adam/add_41AddV2training/Adam/mul_68training/Adam/mul_69*
T0*
_output_shapes	
:�
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
training/Adam/add_42/yConst*
dtype0*
_output_shapes
: *
valueB
 *���3
r
training/Adam/add_42AddV2training/Adam/Sqrt_14training/Adam/add_42/y*
_output_shapes	
:�*
T0
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
training/Adam/mul_72Multraining/Adam/sub_44training/Adam/gradients/AddN_3*
_output_shapes	
:�*
T0
o
training/Adam/add_43AddV2training/Adam/mul_71training/Adam/mul_72*
_output_shapes	
:�*
T0
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
training/Adam/sub_45Subtraining/Adam/sub_45/x training/Adam/ReadVariableOp_117*
_output_shapes
: *
T0
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
training/Adam/clip_by_value_15Maximum&training/Adam/clip_by_value_15/Minimumtraining/Adam/Const_31*
T0*
_output_shapes	
:�
c
training/Adam/Sqrt_15Sqrttraining/Adam/clip_by_value_15*
_output_shapes	
:�*
T0
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
training/Adam/truediv_15RealDivtraining/Adam/mul_75training/Adam/add_45*
T0*
_output_shapes	
:�
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
training/Adam/sub_47/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
v
training/Adam/sub_47Subtraining/Adam/sub_47/x training/Adam/ReadVariableOp_123*
T0*
_output_shapes
: 
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
training/Adam/Square_15Squaretraining/Adam/gradients/AddN_1*
_output_shapes	
:�*
T0
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
&training/Adam/clip_by_value_16/MinimumMinimumtraining/Adam/add_47training/Adam/Const_34*
_output_shapes	
:�*
T0
�
training/Adam/clip_by_value_16Maximum&training/Adam/clip_by_value_16/Minimumtraining/Adam/Const_33*
_output_shapes	
:�*
T0
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
training/Adam/sub_49Sub training/Adam/ReadVariableOp_126training/Adam/truediv_16*
T0*
_output_shapes	
:�
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
training/Adam/sub_50/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
v
training/Adam/sub_50Subtraining/Adam/sub_50/x training/Adam/ReadVariableOp_131*
_output_shapes
: *
T0
�
training/Adam/mul_82Multraining/Adam/sub_504training/Adam/gradients/dense_5/MatMul_grad/MatMul_1*
T0*
_output_shapes
:	�+
s
training/Adam/add_49AddV2training/Adam/mul_81training/Adam/mul_82*
_output_shapes
:	�+*
T0
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
training/Adam/sub_51/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
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
training/Adam/add_52AddV2training/Adam/mul_86training/Adam/mul_87*
T0*
_output_shapes
:+
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
training/Adam/Square_17Square8training/Adam/gradients/dense_5/BiasAdd_grad/BiasAddGrad*
_output_shapes
:+*
T0
o
training/Adam/mul_89Multraining/Adam/sub_54training/Adam/Square_17*
_output_shapes
:+*
T0
n
training/Adam/add_53AddV2training/Adam/mul_88training/Adam/mul_89*
_output_shapes
:+*
T0
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
_
training/VarIsInitializedOpVarIsInitializedOptraining/Adam/m_12_1*
_output_shapes
: 
`
training/VarIsInitializedOp_1VarIsInitializedOptraining/Adam/v_2_1*
_output_shapes
: 
c
training/VarIsInitializedOp_2VarIsInitializedOptraining/Adam/vhat_3_1*
_output_shapes
: 
c
training/VarIsInitializedOp_3VarIsInitializedOptraining/Adam/vhat_9_1*
_output_shapes
: 
d
training/VarIsInitializedOp_4VarIsInitializedOptraining/Adam/vhat_15_1*
_output_shapes
: 
[
training/VarIsInitializedOp_5VarIsInitializedOpdense_2/kernel*
_output_shapes
: 
R
training/VarIsInitializedOp_6VarIsInitializedOptotal*
_output_shapes
: 
`
training/VarIsInitializedOp_7VarIsInitializedOptraining/Adam/m_0_1*
_output_shapes
: 
a
training/VarIsInitializedOp_8VarIsInitializedOptraining/Adam/v_17_1*
_output_shapes
: 
c
training/VarIsInitializedOp_9VarIsInitializedOptraining/Adam/vhat_6_1*
_output_shapes
: 
e
training/VarIsInitializedOp_10VarIsInitializedOptraining/Adam/vhat_12_1*
_output_shapes
: 
]
training/VarIsInitializedOp_11VarIsInitializedOpAdam/iterations*
_output_shapes
: 
a
training/VarIsInitializedOp_12VarIsInitializedOptraining/Adam/m_2_1*
_output_shapes
: 
b
training/VarIsInitializedOp_13VarIsInitializedOptraining/Adam/v_11_1*
_output_shapes
: 
d
training/VarIsInitializedOp_14VarIsInitializedOptraining/Adam/vhat_0_1*
_output_shapes
: 
e
training/VarIsInitializedOp_15VarIsInitializedOptraining/Adam/vhat_14_1*
_output_shapes
: 
a
training/VarIsInitializedOp_16VarIsInitializedOptraining/Adam/v_7_1*
_output_shapes
: 
Z
training/VarIsInitializedOp_17VarIsInitializedOpdense_1/bias*
_output_shapes
: 
o
training/VarIsInitializedOp_18VarIsInitializedOp!batch_normalization_1/moving_mean*
_output_shapes
: 
s
training/VarIsInitializedOp_19VarIsInitializedOp%batch_normalization_1/moving_variance*
_output_shapes
: 
s
training/VarIsInitializedOp_20VarIsInitializedOp%batch_normalization_3/moving_variance*
_output_shapes
: 
h
training/VarIsInitializedOp_21VarIsInitializedOpbatch_normalization_4/beta*
_output_shapes
: 
S
training/VarIsInitializedOp_22VarIsInitializedOpcount*
_output_shapes
: 
a
training/VarIsInitializedOp_23VarIsInitializedOptraining/Adam/m_1_1*
_output_shapes
: 
a
training/VarIsInitializedOp_24VarIsInitializedOptraining/Adam/m_7_1*
_output_shapes
: 
b
training/VarIsInitializedOp_25VarIsInitializedOptraining/Adam/m_11_1*
_output_shapes
: 
a
training/VarIsInitializedOp_26VarIsInitializedOptraining/Adam/v_1_1*
_output_shapes
: 
a
training/VarIsInitializedOp_27VarIsInitializedOptraining/Adam/v_4_1*
_output_shapes
: 
Z
training/VarIsInitializedOp_28VarIsInitializedOpdense_2/bias*
_output_shapes
: 
\
training/VarIsInitializedOp_29VarIsInitializedOpdense_3/kernel*
_output_shapes
: 
\
training/VarIsInitializedOp_30VarIsInitializedOpdense_5/kernel*
_output_shapes
: 
d
training/VarIsInitializedOp_31VarIsInitializedOptraining/Adam/vhat_4_1*
_output_shapes
: 
e
training/VarIsInitializedOp_32VarIsInitializedOptraining/Adam/vhat_10_1*
_output_shapes
: 
e
training/VarIsInitializedOp_33VarIsInitializedOptraining/Adam/vhat_16_1*
_output_shapes
: 
d
training/VarIsInitializedOp_34VarIsInitializedOptraining/Adam/vhat_1_1*
_output_shapes
: 
d
training/VarIsInitializedOp_35VarIsInitializedOptraining/Adam/vhat_7_1*
_output_shapes
: 
i
training/VarIsInitializedOp_36VarIsInitializedOpbatch_normalization_1/gamma*
_output_shapes
: 
a
training/VarIsInitializedOp_37VarIsInitializedOptraining/Adam/m_8_1*
_output_shapes
: 
e
training/VarIsInitializedOp_38VarIsInitializedOptraining/Adam/vhat_13_1*
_output_shapes
: 
a
training/VarIsInitializedOp_39VarIsInitializedOptraining/Adam/m_3_1*
_output_shapes
: 
Y
training/VarIsInitializedOp_40VarIsInitializedOpAdam/beta_2*
_output_shapes
: 
a
training/VarIsInitializedOp_41VarIsInitializedOptraining/Adam/v_6_1*
_output_shapes
: 
o
training/VarIsInitializedOp_42VarIsInitializedOp!batch_normalization_3/moving_mean*
_output_shapes
: 
i
training/VarIsInitializedOp_43VarIsInitializedOpbatch_normalization_4/gamma*
_output_shapes
: 
Z
training/VarIsInitializedOp_44VarIsInitializedOpdense_5/bias*
_output_shapes
: 
b
training/VarIsInitializedOp_45VarIsInitializedOptraining/Adam/m_15_1*
_output_shapes
: 
a
training/VarIsInitializedOp_46VarIsInitializedOptraining/Adam/v_5_1*
_output_shapes
: 
Z
training/VarIsInitializedOp_47VarIsInitializedOpdense_4/bias*
_output_shapes
: 
b
training/VarIsInitializedOp_48VarIsInitializedOptraining/Adam/v_12_1*
_output_shapes
: 
i
training/VarIsInitializedOp_49VarIsInitializedOpbatch_normalization_3/gamma*
_output_shapes
: 
a
training/VarIsInitializedOp_50VarIsInitializedOptraining/Adam/m_4_1*
_output_shapes
: 
b
training/VarIsInitializedOp_51VarIsInitializedOptraining/Adam/m_14_1*
_output_shapes
: 
b
training/VarIsInitializedOp_52VarIsInitializedOptraining/Adam/m_16_1*
_output_shapes
: 
d
training/VarIsInitializedOp_53VarIsInitializedOptraining/Adam/vhat_5_1*
_output_shapes
: 
i
training/VarIsInitializedOp_54VarIsInitializedOpbatch_normalization_2/gamma*
_output_shapes
: 
b
training/VarIsInitializedOp_55VarIsInitializedOptraining/Adam/v_15_1*
_output_shapes
: 
e
training/VarIsInitializedOp_56VarIsInitializedOptraining/Adam/vhat_11_1*
_output_shapes
: 
e
training/VarIsInitializedOp_57VarIsInitializedOptraining/Adam/vhat_17_1*
_output_shapes
: 
d
training/VarIsInitializedOp_58VarIsInitializedOptraining/Adam/vhat_8_1*
_output_shapes
: 
d
training/VarIsInitializedOp_59VarIsInitializedOptraining/Adam/vhat_2_1*
_output_shapes
: 
b
training/VarIsInitializedOp_60VarIsInitializedOptraining/Adam/m_10_1*
_output_shapes
: 
a
training/VarIsInitializedOp_61VarIsInitializedOptraining/Adam/v_8_1*
_output_shapes
: 
b
training/VarIsInitializedOp_62VarIsInitializedOptraining/Adam/v_16_1*
_output_shapes
: 
b
training/VarIsInitializedOp_63VarIsInitializedOptraining/Adam/v_14_1*
_output_shapes
: 
Y
training/VarIsInitializedOp_64VarIsInitializedOpAdam/beta_1*
_output_shapes
: 
a
training/VarIsInitializedOp_65VarIsInitializedOptraining/Adam/m_6_1*
_output_shapes
: 
\
training/VarIsInitializedOp_66VarIsInitializedOpdense_4/kernel*
_output_shapes
: 
h
training/VarIsInitializedOp_67VarIsInitializedOpbatch_normalization_1/beta*
_output_shapes
: 
h
training/VarIsInitializedOp_68VarIsInitializedOpbatch_normalization_2/beta*
_output_shapes
: 
`
training/VarIsInitializedOp_69VarIsInitializedOpAdam/learning_rate*
_output_shapes
: 
X
training/VarIsInitializedOp_70VarIsInitializedOp
Adam/decay*
_output_shapes
: 
a
training/VarIsInitializedOp_71VarIsInitializedOptraining/Adam/m_5_1*
_output_shapes
: 
a
training/VarIsInitializedOp_72VarIsInitializedOptraining/Adam/m_9_1*
_output_shapes
: 
s
training/VarIsInitializedOp_73VarIsInitializedOp%batch_normalization_4/moving_variance*
_output_shapes
: 
a
training/VarIsInitializedOp_74VarIsInitializedOptraining/Adam/v_3_1*
_output_shapes
: 
b
training/VarIsInitializedOp_75VarIsInitializedOptraining/Adam/m_13_1*
_output_shapes
: 
b
training/VarIsInitializedOp_76VarIsInitializedOptraining/Adam/v_10_1*
_output_shapes
: 
a
training/VarIsInitializedOp_77VarIsInitializedOptraining/Adam/v_0_1*
_output_shapes
: 
a
training/VarIsInitializedOp_78VarIsInitializedOptraining/Adam/v_9_1*
_output_shapes
: 
Z
training/VarIsInitializedOp_79VarIsInitializedOpdense_3/bias*
_output_shapes
: 
h
training/VarIsInitializedOp_80VarIsInitializedOpbatch_normalization_3/beta*
_output_shapes
: 
s
training/VarIsInitializedOp_81VarIsInitializedOp%batch_normalization_2/moving_variance*
_output_shapes
: 
o
training/VarIsInitializedOp_82VarIsInitializedOp!batch_normalization_2/moving_mean*
_output_shapes
: 
o
training/VarIsInitializedOp_83VarIsInitializedOp!batch_normalization_4/moving_mean*
_output_shapes
: 
b
training/VarIsInitializedOp_84VarIsInitializedOptraining/Adam/v_13_1*
_output_shapes
: 
\
training/VarIsInitializedOp_85VarIsInitializedOpdense_1/kernel*
_output_shapes
: 
b
training/VarIsInitializedOp_86VarIsInitializedOptraining/Adam/m_17_1*
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
%batch_normalization_1/cond/switch_t:0L
$batch_normalization_1/cond/pred_id:0$batch_normalization_1/cond/pred_id:0P
'batch_normalization_1/batchnorm/add_1:0%batch_normalization_1/cond/Switch_1:1
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
dense_1/Relu:0E
dense_1/Relu:03batch_normalization_1/cond/batchnorm/mul_1/Switch:0L
$batch_normalization_1/cond/pred_id:0$batch_normalization_1/cond/pred_id:0g
'batch_normalization_1/moving_variance:0<batch_normalization_1/cond/batchnorm/ReadVariableOp/Switch:0e
#batch_normalization_1/moving_mean:0>batch_normalization_1/cond/batchnorm/ReadVariableOp_1/Switch:0^
batch_normalization_1/beta:0>batch_normalization_1/cond/batchnorm/ReadVariableOp_2/Switch:0a
batch_normalization_1/gamma:0@batch_normalization_1/cond/batchnorm/mul/ReadVariableOp/Switch:0
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
dense_2/Relu:0^
batch_normalization_2/beta:0>batch_normalization_2/cond/batchnorm/ReadVariableOp_2/Switch:0a
batch_normalization_2/gamma:0@batch_normalization_2/cond/batchnorm/mul/ReadVariableOp/Switch:0E
dense_2/Relu:03batch_normalization_2/cond/batchnorm/mul_1/Switch:0e
#batch_normalization_2/moving_mean:0>batch_normalization_2/cond/batchnorm/ReadVariableOp_1/Switch:0g
'batch_normalization_2/moving_variance:0<batch_normalization_2/cond/batchnorm/ReadVariableOp/Switch:0L
$batch_normalization_2/cond/pred_id:0$batch_normalization_2/cond/pred_id:0
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
dropout_1/cond/switch_f:0?
"batch_normalization_2/cond/Merge:0dropout_1/cond/Switch_1:04
dropout_1/cond/pred_id:0dropout_1/cond/pred_id:0
�
$batch_normalization_3/cond/cond_text$batch_normalization_3/cond/pred_id:0%batch_normalization_3/cond/switch_t:0 *�
'batch_normalization_3/batchnorm/add_1:0
%batch_normalization_3/cond/Switch_1:0
%batch_normalization_3/cond/Switch_1:1
$batch_normalization_3/cond/pred_id:0
%batch_normalization_3/cond/switch_t:0P
'batch_normalization_3/batchnorm/add_1:0%batch_normalization_3/cond/Switch_1:1L
$batch_normalization_3/cond/pred_id:0$batch_normalization_3/cond/pred_id:0
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
batch_normalization_3/beta:0>batch_normalization_3/cond/batchnorm/ReadVariableOp_2/Switch:0e
#batch_normalization_3/moving_mean:0>batch_normalization_3/cond/batchnorm/ReadVariableOp_1/Switch:0g
'batch_normalization_3/moving_variance:0<batch_normalization_3/cond/batchnorm/ReadVariableOp/Switch:0E
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
dense_4/Relu:0E
dense_4/Relu:03batch_normalization_4/cond/batchnorm/mul_1/Switch:0L
$batch_normalization_4/cond/pred_id:0$batch_normalization_4/cond/pred_id:0a
batch_normalization_4/gamma:0@batch_normalization_4/cond/batchnorm/mul/ReadVariableOp/Switch:0^
batch_normalization_4/beta:0>batch_normalization_4/cond/batchnorm/ReadVariableOp_2/Switch:0g
'batch_normalization_4/moving_variance:0<batch_normalization_4/cond/batchnorm/ReadVariableOp/Switch:0e
#batch_normalization_4/moving_mean:0>batch_normalization_4/cond/batchnorm/ReadVariableOp_1/Switch:0"�X
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
training/Adam/vhat_17_1:0training/Adam/vhat_17_1/Assign-training/Adam/vhat_17_1/Read/ReadVariableOp:0(2training/Adam/vhat_17:08I@�       ���	#��۠��A*

val_loss"��?�L3�        )��P	��۠��A*

val_accuracy<w�>��B,       �K"	���۠��A*

loss!�'@ht��       ���	۠��A*

accuracy
S�>��       ��2	��2ݠ��A*

val_loss�n�?\G�"       x=�	��2ݠ��A*

val_accuracy��"?'�MD       ��-	�2ݠ��A*

loss���?�,��       ��2	r�2ݠ��A*

accuracy�?�B�H       ��2	?��ޠ��A*

val_lossd�?;��"       x=�	��ޠ��A*

val_accuracy�#?()Q�       ��-	���ޠ��A*

loss�h?I�n#       ��2	��ޠ��A*

accuracy}9?� g       ��2	rR^ࠫ�A*

val_loss��U?�kg�"       x=�	PS^ࠫ�A*

val_accuracy�'H?��;       ��-	�S^ࠫ�A*

loss�B?��W�       ��2	;T^ࠫ�A*

accuracy��D?+"q�       ��2	k��ᠫ�A*

val_loss�;[?X+��"       x=�	N��ᠫ�A*

val_accuracy�=?<�*�       ��-	���ᠫ�A*

loss (?���       ��2	8��ᠫ�A*

accuracy܄L?l�x       ��2	�㠫�A*

val_loss1�?m�"       x=�	
�㠫�A*

val_accuracy��??����       ��-	��㠫�A*

loss��?����       ��2	�㠫�A*

accuracyy�R?��A       ��2	%u*堫�A*

val_lossZA�?�S�"       x=�	-v*堫�A*

val_accuracytC7?Q(�       ��-	�v*堫�A*

loss�?&�(�       ��2	Kw*堫�A*

accuracy	YW?�iQ�       ��2	Q��栫�A*

val_loss�:|?ɘ��"       x=�	+��栫�A*

val_accuracy��<?b˄�       ��-	���栫�A*

lossД�>G)�       ��2	��栫�A*

accuracy';Y?�
m,       ��2	��u蠫�A*

val_lossmy?���"       x=�	��u蠫�A*

val_accuracy5�W?P �       ��-	��u蠫�A*

lossi��>5g��       ��2	+�u蠫�A*

accuracy܆\?����       ��2	ؽ.꠫�A	*

val_loss��?����"       x=�	'�.꠫�A	*

val_accuracyLW?.�ZP       ��-	��.꠫�A	*

loss(~�>�o=�       ��2	�.꠫�A	*

accuracytc_?_<;       ��2	Y��렫�A
*

val_loss8{,?_y{$"       x=�	G��렫�A
*

val_accuracy>�Q?�c3T       ��-	ɍ�렫�A
*

lossH��>/�>       ��2	?��렫�A
*

accuracya6`?i�       ��2	��g����A*

val_loss��?8��;"       x=�	��g����A*

val_accuracy2dU?3�"M       ��-	��g����A*

loss�0�>K��       ��2	�g����A*

accuracy�b?�VJm       ��2	� W�A*

val_lossuP&?_Q�s"       x=�	�!W�A*

val_accuracy��U?��P       ��-	O"W�A*

loss���>�T��       ��2	�"W�A*

accuracy��a?��,t       ��2	y����A*

val_loss��)?���"       x=�	����A*

val_accuracy�(U?�9       ��-	����A*

loss�ɾ>N���       ��2	����A*

accuracy�c?�vĎ       ��2	������A*

val_loss��'?��"       x=�	������A*

val_accuracyH�T?���!       ��-	ܜ����A*

lossP�>�;�       ��2	������A*

accuracy�<f?�Df       ��2	������A*

val_lossF�`?g�O$"       x=�	������A*

val_accuracy�K?��s�       ��-	2������A*

loss�&�>5�       ��2	������A*

accuracy��d?�#��       ��2	��f����A*

val_loss��?����"       x=�	U�f����A*

val_accuracyR\?��[u       ��-	�f����A*

loss�z�>�ܿ}       ��2	��f����A*

accuracy�g?�ԡ�       ��2	;S���A*

val_loss�e>?%и�"       x=�	�U���A*

val_accuracy�*P?�H]L       ��-	W���A*

lossH2�>_S�       ��2	�W���A*

accuracy��f?���       ��2	좫���A*

val_loss`�?+T"       x=�	����A*

val_accuracy�[?��]       ��-	�����A*

lossUi�>#��       ��2	ۦ����A*

accuracy�h?�.`�       ��2	d p���A*

val_loss
f?^��2"       x=�	)p���A*

val_accuracy"W?�ښ       ��-	up���A*

loss�̞>-�Y       ��2	pp���A*

accuracy��g?!:1       ��2	h���A*

val_loss�?��l-"       x=�	�i���A*

val_accuracylY?���$       ��-	�j���A*

loss��>�/6       ��2	�k���A*

accuracyͳi?S���       ��2	�����A*

val_loss�/?��+L"       x=�	D����A*

val_accuracy"W?�zұ       ��-	.����A*

loss�݉>�lm�       ��2	����A*

accuracy�Dk?��Hc       ��2	v�>���A*

val_lossAx�>���8"       x=�	��>���A*

val_accuracy�k`?��       ��-	!�>���A*

loss�G�>�j�H       ��2	�>���A*

accuracy��j?�k��       ��2	ю
���A*

val_loss�D?_���"       x=�	Ő
���A*

val_accuracy�|[?Z�=�       ��-	ڑ
���A*

loss�.�>���p       ��2	��
���A*

accuracy��k?���       ��2	2����A*

val_lossG�?��"       x=�	����A*

val_accuracyf:]?��Z2       ��-	�����A*

loss�ׇ>$A8       ��2	�����A*

accuracy,Ak?XqB�       ��2	<�����A*

val_lossq�?���?"       x=�	i�����A*

val_accuracyǼ^?��j�       ��-	e�����A*

loss��>�C^�       ��2	.�����A*

accuracyvl?����       ��2	Þ�#���A*

val_loss��	?O���"       x=�	���#���A*

val_accuracy��]?OT.w       ��-	¢�#���A*

loss�=�>>"�       ��2	���#���A*

accuracyb�k?;;��       ��2	�"�'���A*

val_loss��?���L"       x=�	%�'���A*

val_accuracy�]?М��       ��-	&�'���A*

loss�>HU��       ��2	�&�'���A*

accuracyPal?�4)       ��2	L��+���A*

val_lossI�c?��m["       x=�	n��+���A*

val_accuracy��O?��è       ��-	���+���A*

loss��t>���       ��2	Y��+���A*

accuracyŬm?���U       ��2	�3/���A*

val_loss-�N?P'�/"       x=�	��3/���A*

val_accuracy�4M?_P�       ��-	g�3/���A*

loss�+s>���       ��2	�3/���A*

accuracy�jm?��       ��2	D�U2���A*

val_loss�:?�H�"       x=�	�U2���A*

val_accuracy?4V?�:�       ��-	ҧU2���A*

loss�j>3d��       ��2	��U2���A*

accuracyYn?w[f�       ��2	�Fp5���A*

val_loss�
?Ta�"       x=�	kHp5���A*

val_accuracy��b?̣Iy       ��-	ZIp5���A*

loss&s>�{��       ��2	Jp5���A*

accuracy�_m?���       ��2	1�8���A *

val_lossT?����"       x=�	82�8���A *

val_accuracyp�[?Q��y       ��-	�2�8���A *

lossGe>����       ��2	�3�8���A *

accuracy)xn?�n        ��2	e;D<���A!*

val_loss��?��-"       x=�	=D<���A!*

val_accuracyL0`?�熮       ��-	�=D<���A!*

loss-�a>�((�       ��2	�>D<���A!*

accuracy��n?����       ��2	�; @���A"*

val_loss��?�"�"       x=�	�= @���A"*

val_accuracy%�]?�N7        ��-	�> @���A"*

loss��\>�8�       ��2	�? @���A"*

accuracy�No?oeֿ       ��2	}|�C���A#*

val_loss�?H��"       x=�	V~�C���A#*

val_accuracyF�_?����       ��-	^�C���A#*

loss`�V>�$�       ��2	A��C���A#*

accuracyKo?{��       ��2	s�G���A$*

val_loss�u4?��/"       x=�	�t�G���A$*

val_accuracy��V?jg�=       ��-	�u�G���A$*

lossɪb>�P�       ��2	qv�G���A$*

accuracyetn??9��       ��2	�d�K���A%*

val_loss�>/?J7��"       x=�	g�K���A%*

val_accuracy��X?���       ��-	#h�K���A%*

loss�vR>\q�       ��2	�h�K���A%*

accuracy]p?k���       ��2	6�O���A&*

val_lossG9~?d��"       x=�	��O���A&*

val_accuracy�4M?��N       ��-	��O���A&*

lossaZ>*��
       ��2	��O���A&*

accuracy�no?K�&�       ��2	�7hR���A'*

val_loss�%�>�Z�."       x=�	m9hR���A'*

val_accuracy]c?��(       ��-	m:hR���A'*

loss-�A>���`       ��2	`;hR���A'*

accuracytIq?y$�C       ��2	���U���A(*

val_lossu�?�5k "       x=�	 ��U���A(*

val_accuracy�X?�([       ��-	��U���A(*

loss�UD>�!\�       ��2	�U���A(*

accuracy �p?�u2�       ��2	��
Y���A)*

val_loss�3?�)?�"       x=�	e�
Y���A)*

val_accuracyY a?\�y�       ��-	&�
Y���A)*

lossIQI>���       ��2	��
Y���A)*

accuracyixp?�B �       ��2	�Z�\���A**

val_lossCz?�x"       x=�	�\�\���A**

val_accuracy�z`?rȲ       ��-	�]�\���A**

loss�3>�*�m       ��2	V^�\���A**

accuracy?or?H˸       ��2	̤`���A+*

val_loss�w7?|� �"       x=�	�ͤ`���A+*

val_accuracy�Y?�y��       ��-	�Τ`���A+*

loss�'D>����       ��2	)Ϥ`���A+*

accuracy��p?���P       ��2	1��c���A,*

val_loss��=?��7�"       x=�	���c���A,*

val_accuracy��V?�ڰF       ��-	ß�c���A,*

loss�<>�v�       ��2	���c���A,*

accuracy�8q?�e       ��2	Zg���A-*

val_loss(�?o�L&"       x=�	�Zg���A-*

val_accuracyN`?8,34       ��-	�Zg���A-*

loss�!;>aW�g       ��2	GZg���A-*

accuracy	�q?�ݖ�       ��2	�38k���A.*

val_loss��@?��J�"       x=�	v58k���A.*

val_accuracy(�[?w�:�       ��-	q68k���A.*

loss��=>m�,�       ��2	K78k���A.*

accuracy�Aq?�w�       ��2	���n���A/*

val_lossO)?]�U1"       x=�	���n���A/*

val_accuracy�a?�"{2       ��-	��n���A/*

loss��@><��e       ��2	��n���A/*

accuracy,^q?�?3J       ��2	�H"r���A0*

val_lossC�/?/Ҕe"       x=�	J"r���A0*

val_accuracyZX?+ؽ       ��-	K"r���A0*

lossQn4>�"W       ��2	<L"r���A0*

accuracyYMr?S��[       ��2	۪5u���A1*

val_loss1��>��"       x=�	��5u���A1*

val_accuracy<f?d�l�       ��-	��5u���A1*

lossL4>��*]       ��2	h�5u���A1*

accuracyj\r?]��       ��2	bK=x���A2*

val_loss��?F��"       x=�	�L=x���A2*

val_accuracy��a?�mt:       ��-	�M=x���A2*

loss��'>w��       ��2	"N=x���A2*

accuracy�es?�y�       ��2	J��{���A3*

val_loss��'?)�>o"       x=�	B��{���A3*

val_accuracyV�^?��}       ��-	��{���A3*

loss�7>G�P�       ��2	���{���A3*

accuracy�r?��       ��2	ɯ~���A4*

val_lossN�?��q "       x=�	qʯ~���A4*

val_accuracyzc?W�l�       ��-	2˯~���A4*

loss�V'>k퉮       ��2	�˯~���A4*

accuracy}ms?0�d       ��2	��Ɓ���A5*

val_loss��?��s?"       x=�	�Ɓ���A5*

val_accuracy��]?Yv�8       ��-	��Ɓ���A5*

loss��.>�2�       ��2	��Ɓ���A5*

accuracy��r?�B�       ��2	�B&����A6*

val_loss��?�-�"       x=�	�C&����A6*

val_accuracy��b?O�-�       ��-	�D&����A6*

loss{�$>"��_       ��2	=E&����A6*

accuracyV�s?�;�?       ��2	�#Q����A7*

val_loss�~?ܝ]"       x=�	�%Q����A7*

val_accuracy�Rc?b�|�       ��-	l&Q����A7*

loss�o$>ƿ�O       ��2	 'Q����A7*

accuracy�Zs?_�e       ��2	h�����A8*

val_loss�i%?z�/="       x=�	�����A8*

val_accuracys
^?���       ��-	ɲ����A8*

loss��">-�       ��2	������A8*

accuracy<�s?ގ�       ��2	ݓH����A9*

val_loss.O�>�)m"       x=�	��H����A9*

val_accuracy�b?���       ��-	 �H����A9*

loss��$>�gq�       ��2	��H����A9*

accuracys?��o       ��2	�7̒���A:*

val_loss�?%�c�"       x=�	d9̒���A:*

val_accuracy2�^?vPg�       ��-	G:̒���A:*

loss��'>c���       ��2	;̒���A:*

accuracySs?Z�iT       ��2	�l����A;*

val_loss�&?n�5�"       x=�	��l����A;*

val_accuracy"md?Hr��       ��-	��l����A;*

loss�N>���       ��2	9�l����A;*

accuracyft?��       ��2	_󈙡��A<*

val_loss��?�%��"       x=�	U������A<*

val_accuracys�b?w        ��-	8������A<*

lossN!>3^�Y       ��2	�������A<*

accuracyus?�0Ae       ��2	�Mݜ���A=*

val_loss�%$?�?�"       x=�	+Oݜ���A=*

val_accuracy��_?lKj�       ��-	�Oݜ���A=*

loss��>Wq�D       ��2	�Pݜ���A=*

accuracy��s?��       ��2	-c����A>*

val_loss��?�cM"       x=�	�.c����A>*

val_accuracyeb?ǒD       ��-	]/c����A>*

loss�P>�6�w       ��2	0c����A>*

accuracy��t?t�rD       ��2	������A?*

val_loss�?��o"       x=�	������A?*

val_accuracy�T^?�Tv       ��-	������A?*

lossP>x_       ��2	o�����A?*

accuracy�+t?[�kH       ��2	v�զ���A@*

val_loss>0?�+t�"       x=�	z�զ���A@*

val_accuracy�]?�ۖ       ��-	��զ���A@*

loss��>��U�       ��2	u�զ���A@*

accuracyLBt?�&�       ��2	������AA*

val_lossb?�W��"       x=�	������AA*

val_accuracy��g?txYR       ��-	������AA*

loss�0>�d--       ��2	>�����AA*

accuracy�mt?���Q       ��2	��3����AB*

val_loss�??0��2"       x=�	`�3����AB*

val_accuracy?�c?�ӯ�       ��-	%�3����AB*

loss�>�_�       ��2	��3����AB*

accuracy?St?Ld?       ��2	 �����AC*

val_lossU�?*w"       x=�	�����AC*

val_accuracy�b?d,�?       ��-	������AC*

lossH�>k��       ��2	������AC*

accuracyFt?�n�7       ��2	�᳡��AD*

val_loss�>?�[�"       x=�	*᳡��AD*

val_accuracy�;a?��H�       ��-	᳡��AD*

loss��>�-K�       ��2	�᳡��AD*

accuracyaqt?uM9z       ��2	�x����AE*

val_lossz0?BJ^1"       x=�	Fz����AE*

val_accuracy�\?��t'       ��-	%{����AE*

lossٟ>;��       ��2	�{����AE*

accuracy�<t?>�p       ��2	$B]����AF*

val_loss��> Z�3"       x=�	�C]����AF*

val_accuracy�nh?RX�+       ��-	�D]����AF*

loss�)>.k��       ��2	$E]����AF*

accuracy��t?�f��       ��2	֫�����AG*

val_loss `?W}"       x=�	q������AG*

val_accuracy��b?��4       ��-	`������AG*

loss
7>�6	       ��2	)������AG*

accuracy��u?x�       ��2	�!�����AH*

val_loss30?�p0"       x=�	=#�����AH*

val_accuracy�O[?hu�k       ��-	$�����AH*

loss��>���e       ��2	�$�����AH*

accuracyët?	�?o       ��2	D�ġ��AI*

val_loss�# ?l_"       x=�	��ġ��AI*

val_accuracyY a?��5       ��-	f�ġ��AI*

loss�\>���1       ��2		�ġ��AI*

accuracy�+u?�� �       ��2	Xhǡ��AJ*

val_loss��5?���/"       x=�	� hǡ��AJ*

val_accuracy�]?4���       ��-	�!hǡ��AJ*

lossX! >�Sz       ��2	}"hǡ��AJ*

accuracy�Ov?��b�       ��2	+m�ʡ��AK*

val_loss�?Va9"       x=�	�n�ʡ��AK*

val_accuracy�@d?��0e       ��-	�o�ʡ��AK*

loss >����       ��2	Lp�ʡ��AK*

accuracyּu?�*��       ��2	]�͡��AL*

val_loss�V?!�0"       x=�	[^�͡��AL*

val_accuracy�a?�|ɳ       ��-	_�͡��AL*

lossN�>̰u       ��2	�_�͡��AL*

accuracy��t?�Z       ��2	��С��AM*

val_loss�?�/@m"       x=�	r��С��AM*

val_accuracy��b?�Y�       ��-	'��С��AM*

loss�g>CX�       ��2	�С��AM*

accuracy�u?�8       ��2	9�ӡ��AN*

val_loss۹'?O�"       x=�	�:�ӡ��AN*

val_accuracy��b?^�(       ��-	<�ӡ��AN*

loss�%>���V       ��2	�<�ӡ��AN*

accuracy4fu?$ɩ       ��2	�ס��AO*

val_loss�)
?�{�"       x=�	��ס��AO*

val_accuracy�*f?�vd�       ��-	��ס��AO*

loss���=!'&~       ��2	g�ס��AO*

accuracy� v?Vb�J       ��2	�^�ڡ��AP*

val_loss�f-?���r"       x=�	�`�ڡ��AP*

val_accuracyF�_?�^�v       ��-	�a�ڡ��AP*

loss�}>[��       ��2	bb�ڡ��AP*

accuracycsu?�Զ�       ��2	���ݡ��AQ*

val_loss3	?���?"       x=�	T��ݡ��AQ*

val_accuracy��e?ߤ��       ��-	?��ݡ��AQ*

loss�:>��
�       ��2	��ݡ��AQ*

accuracy�u?�w,       ��2	S��࡫�AR*

val_loss�|�>v��"       x=�	ȴ�࡫�AR*

val_accuracyM\i?��&�       ��-	���࡫�AR*

lossù>':�g       ��2	_��࡫�AR*

accuracy��u?�ʿ{       ��2	iW>䡫�AS*

val_loss��?g->�"       x=�	�X>䡫�AS*

val_accuracy�Rc?�*�       ��-	�Y>䡫�AS*

loss��>7��f       ��2	KZ>䡫�AS*

accuracy�u?�{�       ��2	]��硫�AT*

val_loss�?N��v"       x=�	�硫�AT*

val_accuracy�b?�Pl�       ��-	�硫�AT*

lossf��=B�       ��2	��硫�AT*

accuracy��v?Qbl       ��2	��ꡫ�AU*

val_loss�L?8��"       x=�	s��ꡫ�AU*

val_accuracyB�X?��W       ��-	#��ꡫ�AU*

loss�>�u       ��2	���ꡫ�AU*

accuracycsu?�PO	       ��2	k������AV*

val_lossw-?S��"       x=�	�������AV*

val_accuracy�E^?�Ѳ�       ��-	̺�����AV*

loss�>wOJ�       ��2	�������AV*

accuracy��u?�t       ��2	?���AW*

val_loss"�!?���"       x=�	���AW*

val_accuracy��b?�ע:       ��-	����AW*

lossa>��       ��2	:���AW*

accuracyQ�u?�ۼ�       ��2	������AX*

val_loss\?�^2�"       x=�	2�����AX*

val_accuracy��X?�w��       ��-	�����AX*

loss��=�q0�       ��2	������AX*

accuracy��v?�{,�       ��2	��(����AY*

val_lossd�	?G��"       x=�	��(����AY*

val_accuracy}�e?��%�       ��-	��(����AY*

loss���=�       ��2	K�(����AY*

accuracy��v?�ð       ��2	IO����AZ*

val_loss ;3?+"       x=�	�JO����AZ*

val_accuracyf:]?:��[       ��-	fKO����AZ*

loss3O�=>ȵ	       ��2	LO����AZ*

accuracy��v?���       ��2	��h����A[*

val_loss�?��"       x=�		�h����A[*

val_accuracyP�b?oc�       ��-	��h����A[*

loss_�=Ӡ�h       ��2	H�h����A[*

accuracy?7v?�`�       ��2	�ۍ ���A\*

val_loss�q?33sb"       x=�	ݍ ���A\*

val_accuracy%Vb?���       ��-	�ލ ���A\*

loss>��=Jыc       ��2	]ߍ ���A\*

accuracy�v?����       ��2	5����A]*

val_loss_�?�k."       x=�	E����A]*

val_accuracym8b?�ސ�       ��-	E�����A]*

loss���=�Ď�       ��2	�����A]*

accuracy:�v?�o<       ��2	�JK���A^*

val_loss��>�ˇ�"       x=�	�MK���A^*

val_accuracy�i?���       ��-	�NK���A^*

loss|��=B7�       ��2	�OK���A^*

accuracyCWv?�>p`       ��2	FB�
���A_*

val_lossu� ?���x"       x=�	�D�
���A_*

val_accuracy��`?
x�}       ��-	�E�
���A_*

loss�8>�1       ��2	NF�
���A_*

accuracy��u?>�       ��2	��b���A`*

val_loss��?O���"       x=�	8�b���A`*

val_accuracyI�f? ֪�       ��-	U�b���A`*

loss��=�u�       ��2	3�b���A`*

accuracy�`v?׼/       ��2	�m����Aa*

val_loss���>�X^�"       x=�	p����Aa*

val_accuracy)ki?e?#�       ��-	qq����Aa*

loss:��=j��       ��2	�r����Aa*

accuracy��v?M��s       ��2	�k����Ab*

val_loss-^m?7*��"       x=�	�m����Ab*

val_accuracy�T?eǓ9       ��-	�n����Ab*

loss��=�A.`       ��2	�o����Ab*

accuracyTfv?�J��       ��2	4����Ac*

val_losstW(?�ߦ"       x=�	�5����Ac*

val_accuracy�a?7 ��       ��-	z6����Ac*

loss���=�ۤ       ��2	.7����Ac*

accuracy��v?�
�<