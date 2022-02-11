import typing
from typing import Any, Optional, Union, List, Tuple, get_origin, get_args, Generic, TypeVar, Callable
from enum import Enum
from functools import wraps
import sys
assert sys.version_info.major==3
import inspect
from inspect import signature
import warnings

def value_matches_generic_type(value,real_type,exhaustive_check:bool=False)->bool:
    generic_origin=get_origin(real_type)

    if generic_origin is Union:
        return isinstance(value,get_args(real_type))
    elif generic_origin is tuple:
        for i,t in enumerate(get_args(real_type)):
            if not isinstance(value[i],t):
                return False
        return True
    elif generic_origin is list:
        if len(value)>0:
            if exhaustive_check:
                list_item_type=get_args(real_type)
                for i in range(0,len(value)):
                    if not value_matches_type(value[i],*list_item_type):
                        msg=f"mismatched element type {type(value[i])} at index {i} in typing.List[{list_item_type}]"
                        raise TypeError(msg)

                return True
            else:
                return value_matches_type(value[0],*get_args(real_type))

    return False

def value_matches_type(value,real_type,exhaustive_check:bool=False)->bool:
    return type(value)==real_type or real_type==Any or value_matches_generic_type(value,real_type,exhaustive_check=exhaustive_check)

def type_check(really:bool=True,check_fn:Callable[...,bool]=None,exhaustive_check:bool=False):
    if really:
        def assert_arg_types(fn):
            fn_signature=signature(fn)
            parameter_items=fn_signature.parameters.items()

            for index,(k,p) in enumerate(parameter_items):
                if p.annotation is None:
                    raise RuntimeError("everything needs to be annotated!!!!!")

                if p.default != p.empty:
                    if not value_matches_type(p.default,p.annotation):
                        msg=f"default argument has incompatible type: {p}"
                        raise TypeError(msg)

            @wraps(fn)
            def asserted_fn(*args,**kwargs):
                rest_kwargs=None
                for index,(kw,p) in enumerate(parameter_items):
                    # for <positional or keyword> arguments, this code allows them to fall through to the <keyword> checking code if they have been provided as <keyword> argument, instead of <positional> argument
                    if p.kind in (p.POSITIONAL_OR_KEYWORD, p.POSITIONAL_ONLY) and index<len(args):
                        if not value_matches_type(args[index],p.annotation, exhaustive_check=exhaustive_check):
                            arg_str=f'"{args[index]}"' if type(args[index])==str else args[index]

                            args_type=type(args[index])
                            if args_type is tuple:
                                tuple_types=[type(args[index][i]) for i in range(len(args[index]))]
                                args_type=typing.Tuple[tuple(tuple_types)]
                            elif args_type is list:
                                args_type=typing.List[type(args[index][0])]

                            msg=f"positional argument has incompatible type: {p.name} of type {p.annotation} = {arg_str} of type {args_type}"
                            raise TypeError(msg)

                    elif p.kind in (p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD):
                        if not kw in kwargs:
                            if p.default == p.empty:
                                msg=f"missing keyword argument without default argument {k}"
                                raise RuntimeError(msg)
                            else:
                                # not otherwise specified keyword argument that has default value to be used instead is fine
                                continue

                        if not value_matches_type(kwargs[kw],p.annotation, exhaustive_check=exhaustive_check):
                            kwargs_str=f'"{kwargs[kw]}"' if type(kwargs[kw])==str else kwargs[kw]

                            kwargs_type=type(kwargs[kw])
                            if kwargs_type is tuple:
                                tuple_types=[type(kwargs[kw][i]) for i in range(len(kwargs[kw]))]
                                kwargs_type=typing.Tuple[tuple(tuple_types)]
                            elif kwargs_type is list:
                                kwargs_type=typing.List[type(kwargs[kw][0])]

                            print(type(kwargs[kw]),get_origin(p.annotation))
                            msg=f"keyword argument has incompatible type: {p.name} of type {p.annotation} = {kwargs_str} of type {kwargs_type})"
                            raise TypeError(msg)

                    # other positional args (p.kind==p.VAR_POSITIONAL) and keyword args (p.kind==p.VAR_KEYWORD) will just be forwarded through this code

                if not check_fn is None:
                    if not check_fn(*args,**kwargs,has_result=False,result=None):
                        raise RuntimeError("pre-call invariant did not hold")

                result=fn(*args,**kwargs)

                if not value_matches_type(result,fn_signature.return_annotation, exhaustive_check=exhaustive_check):
                    raise TypeError(f"return value does not match type {return_type}")

                if not check_fn is None:
                    if not check_fn(*args,**kwargs,has_result=True,result=result):
                        raise RuntimeError("post-call invariant did not hold")

                return result

            return asserted_fn
        return assert_arg_types
    else:
        if not check_fn is None:
            msg="specified a check_fn for a function that will not be type-checked! the check_fn will not be executed!"
            warnings.warn(msg,Warning)

        def thin(fn):
            return fn
        return thin
    
if __name__=="__main__":
    #@type_check(True,check_fn=lambda p0,p1,_a,*args,a,b,c,has_result,result,**kwargs:a>0 if has_result else True)
    #def somefn(pos0:int,pos1:str,_a:float=3.2,*args,a:int=3,b:Any,c:Optional[int]=None,**kwargs)->int:
    #    for i in range(0,10000):
    #        a=4*5+a%23
    #    return a*b
    #    
    #for i in range(0,10000):
    #    a=somefn(4,"3.2",3.14,[123123,123,123,12312,3123,123],a=5,b=4,c=None,peter_parker="spiderman")

    @type_check(True,check_fn=lambda pos0,has_result,result:len(pos0)>=2,exhaustive_check=False)
    def somefn2(pos0:List[float])->float:
        return pos0[0]*pos0[1]

    warnings.simplefilter("ignore")
    
    for i in range(0,1000000):
        a=somefn2(pos0=[2.,3.])