import matplotlib
matplotlib.use("agg")

import sys
import numpy
import hist
import pylab
import time
import copy
import inspect
import collections
import itertools



######## autograd for auto diff
try:
    MAUTOGRAD = True
    import autograd
    from autograd import jacobian, primitive
    import autograd.numpy as anumpy
except ImportError:
    MAUTOGRAD = False
###################################

######## scipy.optimize
try:
    MSCIPY = True
    import scipy.optimize
except ImportError:
    MSCIPY = False
###################################
    
### iminuit package

try:
    MIMINUIT = True
    import iminuit
except ImportError:
    MIMINUIT = False


######## nlopt minimization package
## Supported minim functions:
try:
    MNLOPT = True
    import nlopt
except ImportError:
    MNLOPT = False


## metropolis hastings + nuts (HMC) implementation
try:
    MSAMPYL = True
    import sampyl
except ImportError:
    MSAMPYL = False

try:
    MEMCEE = True
    import emcee
except ImportError:
    MEMCEE = False
###################################

#### TODO ... ADD MORE ALGORITHMS... ADAM/ADAGRAD etc



class m(object):
  
  def __init__(self, minim_function, param_names, structure_init, additional_args=[], additional_kwargs=dict(), minimizer_kwargs=dict(), jac_minim_function=None, bounds=dict(), callback=None, callback_args=[], callback_kwargs=dict()):

    self.minim_function=minim_function

    #print self.minim_function

    self.argspec_original=inspect.getargspec(self.minim_function)
    #print "added function with argspect ", self.argspec_original

    self.param_names=param_names
    self.dim=len(self.param_names)
    self.minim_function_jacobian=jac_minim_function

    if(MAUTOGRAD):
      if(self.minim_function_jacobian is not None):
        ## handing over a jacobian ... so it is a normal jacobian without primitive decorator

        ## add *primitive* decorator to define autograds jacobian
        self.minim_function=primitive(self.minim_function)
        ## calculates the jacobian correctly

        def makejacobian_fn(ans, *a, **b):
          def gradient_product(g):
              #print g
              #print x
              return (g*self.minim_function_jacobian(*a, **b).T).sum(axis=1).T
          return gradient_product

        self.minim_function.defgrad(makejacobian_fn)

      #lse:              
      #  scipy_jacobian=jacobian(self.minim_function)
    else:
      if(self.minim_function_jacobian is not None):
        print "Defined a jacobian function but autograd is not installed. Requires Autograd to work, please install autograd if you want the derivative!!!"
        exit(-1)

    # vector containing all calls to optimize and corresponding results
    self.minimization_histories=[]

    ## try auto diff if autograd is installed and jacobian not defined
    

    self.bounds=collections.OrderedDict()
    for pname in self.param_names:
      if(pname in bounds.keys()):
        self.bounds[pname]=bounds[pname]
      else:
        self.bounds[pname]=(-numpy.inf, numpy.inf)

    ## define function structure here...

    init_parameter_values_single_vec=[]

    self.arg_structure=[]
    arg_count=0

    
    #self.current_logged_parameters=[]
    
    ########### find out what the argument structure of the function is .. in the end convert everything to a multidimensional argument

    num_vector_args=0
    num_scalar_args=0
    self.single_vector_arg=False

    for par_i in structure_init:
        if(type(par_i)==numpy.ndarray or type(par_i)==list):
          ## the argument par_i is a vector of individual arguments
          self.arg_structure.append( (len(par_i), "vector") )

          for i in range(len(par_i)):
            init_parameter_values_single_vec.append(par_i[i])

          arg_count+=len(par_i)
          num_vector_args+=1



        else:
          self.arg_structure.append( (1, "scalar"))

          init_parameter_values_single_vec.append(par_i)

          arg_count+=1
          num_scalar_args+=1



    ## the paramater vec that holds initial values .. will be updated after each iteration to the best values from the last one
    self.init_parameter_values_single_vec=numpy.array(init_parameter_values_single_vec)

    if(num_vector_args==1 and num_scalar_args==0):
      self.single_vector_arg=True
    #elif(num_vector_args==0 and num_scalar_args==len(self.param_names)):
    #  self.pure_scalar_args=True

    if(arg_count!=len(self.param_names)):
      print "Error in fn def .."
      print "Argument Count for the function (%d) and the initial parameter structure (%d) does not match" % (len(self.param_names), arg_count)
      exit(-1)

    ## callback funtionality
    self.callback=callback
    self.callback_args=callback_args
    self.callback_kwargs=callback_kwargs

    self.additional_args=additional_args
    self.additional_kwargs=additional_kwargs
    self.minimzer_kwargs=minimizer_kwargs



    self.optimization_results=[]
    self.cur_step_params=[]
    self.cur_step_fnvals=[]
    self.cur_niter=0

    ## saves everything from every eval
    self.total_parvec=[]
    self.total_fnvals=[]

    self.is_optimizing=False

    ######### MINIM DEFS ...
    ###### each method comes with an interface and single argument/multi argument option

    self.minim_defs=dict()

    ### SCIPY

    scipy_methods=["BFGS", "CG", "COBYLA", "dogleg", "L-BFGS-B", "Nelder-Mead", "Newton-CG", "Powell", "SLSQP", "TNC", "trust-ncg"]
    for method in scipy_methods:
      self.minim_defs[method]=("scipy", "single")

    ## IMINUIT

    minuit_methods=["migrad", "simplex"]

    for method in minuit_methods:
      self.minim_defs[method]=("minuit", "multi")

    ### sampyl_METROPOLIS / sampyl_HMC (NUTS)
    
    sampyl_methods=["sampyl_metropolis", "sampyl_nuts"]
    for method in sampyl_methods:
      self.minim_defs[method]=("sampyl", "multi")

    self.minim_defs["emcee"]=("emcee", "single")


  def make_multi(self, fn):

    def multi_fn(*x):
      return fn(anumpy.array(x))

    return multi_fn

  def get_wrapped_fn(self, argument_type="single", check_bounds_and_return_neg_inf=False, force_no_gradient=False, log_steps=False, flip_sign=False):

    ## definition of f(x)

    def f(x):

      tbef=time.time()
      new_arglist=[]

      ## we already have one single vectorized argument .. just copy it
      if(self.single_vector_arg):

        new_arglist.append(x)
      else:

        cur_index=0

        for a in self.arg_structure:
          #print cur_index
          if(a[1]=="scalar"):
            new_arglist.append(x[cur_index])
          else:
            for b in c:
              new_arglist.append(x[cur_index:cur_index+a[0]])

          cur_index+=a[0]


      ## bounds check ... used for sampling algorithms only....

        if(check_bounds_and_return_neg_inf):
          for p_ind, p in enumerate(self.param_names):
            if( (x[p_ind]+1e-12)<self.bounds[p][0] or (x[p_ind]-1e-12)>self.bounds[p][1]):
              if(flip_sign):
                print "breakout"
                return -numpy.inf
              else:
                return numpy.inf
      

      return_val=self.minim_function(*(new_arglist+self.additional_args), **self.additional_kwargs)
      self.cur_niter+=1

      if(type(x)==list or type(x)==numpy.ndarray):


        ## add the negative function value first, before flipping the sign
        if(self.is_optimizing and log_steps):
          self.cur_step_params.append(copy.copy(x))
          self.cur_step_fnvals.append(return_val)
          print "LOGGING"
        if(self.callback is not None):

          self.callback(*self.callback_args, **self.callback_kwargs)
      
      #print "overal outer .... ", time.time()-tbef

      #print new_arglist


      ## used for samplers!
      if(flip_sign):
        return_val*=-1.0

      return return_val
    

    gradient_f=None
    hessian_f=None
    if(argument_type=="single"):

      return f#, gradient_f, hessian_f
    elif(argument_type=="multi"):
      
      return self.make_multi(f)
      #def f_multi(*x):
      #  return f(x)
      #if(MAUTOGRAD):
      #  if(force_no_gradient == False):
      #    gradient_f=jacobian(f_multi)
      #    hessian_f=jacobian(jacobian(f))
      #return f_multi#, gradient_f, hessian_f

###########################################
  
  



  def get_constrained_functions(self, wrapped_fn, constraints, jac=None, profile_method="L-BFGS-B", argument_type="single"):

      ### for every combination of fixed pars, create its own function .. also contianing its own profile function + derivative

      ## constr[0] contains profile indices, constr[1] is the actual list of constraints.....

      #print wrapped_fn
      #print constraints

      
      minim_mask=[]

      ## constraints[1] is representative of all constraints, since only the fixed values change
      for opts in constraints[1]:
        if(opts[0]=="fix"):
          minim_mask.append(False)
        elif(opts[0]=="profile"):
          minim_mask.append(False)
        else:
          minim_mask.extend(len(numpy.arange(opts[2][0], opts[2][1]))*[True])

      minim_mask=numpy.array(minim_mask)

      def return_profile_result(fixed_pars=dict()):

        ## here it is the opposite .. we want to minimize in the profiling dimensions .. so wrapped fn
        ## fixed_pars is dictionary with fix_n0=val" .. etc

        #def inner_profile_fn(x):
        #  new_vals=[]
        #  wrapped_fn()

        ########## FIXME
        ########## initilize based on latest results/closest LLH?! -> fast start parameters for profiling
        #print "--------------------inside profile functions............"
        init_pars=[]

        new_kwargs=dict()
        new_kwargs["log_steps"]=False
        new_kwargs["method"]=profile_method

        for k in fixed_pars.keys():
          new_kwargs[k]=fixed_pars[k]

        #print new_kwargs
        result=self.optimize(**new_kwargs)

        #print "profiling result...", result["pars_vec"]

        return result["pars_vec"]

      ############

      

      def make_constraint_fn(constraint, prof_indices):

        
        
        def fn_constrained(x):
          #new_vec=anumpy.zeros(len(self.total_param_list))
          #new_vec[self.fixed_mask]=self.fixed_vals

          #print "constrained parameter vec..."
          #print y
          #print "---------"

          y_index=0
          fixed_index=0
          new_list=[]

          ## first check for profiling .. do the minimization here
          profile_results=None

          if(len(prof_indices)>0):

            ## define fixed values here ... combination of y + REAL fixed values
            fixed_values=dict()
            val_counter=0

            ## only provide fixed values for fixed and throughgoing parameters
            for opts in constraint:
              if(opts[0]=="fix"):
                fixed_values["fix_"+self.param_names[opts[1][0]]]=opts[1][1]
              elif(opts[0]=="throughgoing"):
                #print opts[1]
                this_y_indices=numpy.arange(opts[1][0], opts[1][1])

                for cur_ind, further_indices in enumerate(numpy.arange(opts[2][0], opts[2][1])):
                  #print "further indies .. ", further_indices
                  fixed_values["fix_"+ self.param_names[further_indices]]=this_y_indices[cur_ind]

                ## provide real indices here
              
            
            profile_results=return_profile_result(fixed_pars=fixed_values)
            #print "RESULTS OF PROFILE: ", profile_results

          for opts in constraint:
              
              if(opts[0]=="throughgoing"):
                #print "through.."
                new_list.append( anumpy.array(x[opts[1][0]:opts[1][1]]))
              elif(opts[0]=="fix"):
                #print "fix.."
                new_list.append(anumpy.array([opts[1][1]]))
              elif(opts[0]=="profile"):
                #print "profile..."
                new_list.append(anumpy.array([profile_results[opts[1]]]))
          #print "newlist ", new_list
          #print new_list
          #print wrapped_fn
          return wrapped_fn(anumpy.concatenate(new_list))

        def fn_multi(*x):

          return fn_constrained(x)

        if(argument_type=="single"):
          return fn_constrained
        else:
          return fn_multi


      profile_indices=constraints[0]

      constrained_fns=[]

      for constr in constraints[1:]:
        

        constrained_fn=make_constraint_fn(constr, profile_indices)
        constrained_fns.append(constrained_fn)

      return constrained_fns, minim_mask


  def generate_constraints_list(self, **kwargs):

    #this_profile_args=[]
    this_fixed_args_unsorted=[]#collections.OrderedDict()

    fixed_names=[]
    fixed_indices=[]
    this_profiling_indices=[]



    ## ordered dictionary to hold all the combinations for the fixing_options
    this_param_constraints=[]

    for k in kwargs.keys():
      #print k
      if("profile_" in k):
        argname=k[8:]
        #this_profile_args.append(argname)
        profile_index=numpy.where(argname==self.param_names)[0][0]
        this_profiling_indices.append(profile_index)
      if("fix_" in k):
        fix_vals=kwargs[k]
        if(type(fix_vals)!=list):
          fix_vals=[fix_vals]
        argname=k[4:]

        if(argname not in self.param_names):
          print argname, "not found in params .. , cannot fix it"
          exit(-1)
        fixed_index=numpy.where(argname==self.param_names)[0][0]
        fixed_indices.append(fixed_index)
        fixed_names.append(argname)
        this_fixed_args_unsorted.append(fix_vals)

    this_profiling_indices=numpy.array(this_profiling_indices)[numpy.argsort(this_profiling_indices)]

    #print "FIXED NAMES", fixed_names

    ## sorting by indices in the overall arg array
    sorta=numpy.argsort(fixed_indices)

    sorted_fixed_args=[]

    
    ## genreate all combinatorics for the fixed arguments
    for product_arg in itertools.product(*this_fixed_args_unsorted):
      
      sorted_fixed_args.append(numpy.array(product_arg)[sorta])

    
    ## the sorted indices of the fixed args
    sorted_fixed_indices=numpy.array(fixed_indices, dtype=numpy.int)[sorta]

    ## the sorted names of the fixed args
    fixed_names=numpy.array(fixed_names)[sorta]

    this_param_constraints.append(this_profiling_indices)
    for combi_index, _ in enumerate(sorted_fixed_args):
      this_param_constraints.append([])

    last_used_index=-1
    fixed_counter=0
    new_par_counter=0
    profile_counter=0

    #print fixed_names
    #print this_profiling_indices

    for ind, parname in enumerate(self.param_names):
        if(ind in sorted_fixed_indices):

            ### ahh we have a fixed index
            if(ind != last_used_index+1):
                ## add previous unfixed range aswell
                #print "adding throughoing ", ind, last_used_index
                diff=ind-last_used_index-1
                for combi_index, fix_val_combi in enumerate(sorted_fixed_args):
                  this_param_constraints[combi_index+1].append( ("throughgoing", (new_par_counter, new_par_counter+diff), (last_used_index+1, last_used_index+1+diff)))
                new_par_counter+=diff

            for combi_index, fix_val_combi in enumerate(sorted_fixed_args):
              this_param_constraints[combi_index+1].append( ("fix", (ind, fix_val_combi[fixed_counter])))

            last_used_index=ind
            fixed_counter+=1
        elif(ind in this_profiling_indices):
          for combi_index, fix_val_combi in enumerate(sorted_fixed_args):
            this_param_constraints[combi_index+1].append(("profile", profile_counter))
          profile_counter+=1

          last_used_index=ind
    

    new_diff=len(self.param_names)-1-last_used_index

    if(last_used_index!=len(self.param_names)-1):
        ## the last param was not fixed .... so add list of other params
        for combi_index, fix_val_combi in enumerate(sorted_fixed_args):
          this_param_constraints[combi_index+1].append( ("throughgoing", (new_par_counter, new_par_counter+new_diff), (last_used_index+1, last_used_index+1+new_diff)))


    return this_param_constraints

  def optimize(self, **kwargs):
    method="L-BFGS-B"

    if(kwargs.has_key("method")):
      method=kwargs["method"]

    if(type(method)==str):
      method=list(method)

    best_result=None
    for m in method:

      new_kwargs=copy.deepcopy(kwargs)

      if(type(m)==tuple):
        new_kwargs["method"]=m[0]
        minimizer_kwargs=dict()
        for it in m[1:]:
          if(type(it)==int):
            minimizer_kwargs["niter"]=it
          elif(type(it)==dict):
            for vdict_key in it.keys():
              minimizer_kwargs[vdict_key]=it[vdict_key]
        new_kwargs["minimizer_kwargs"]=minimizer_kwargs
      else:
        new_kwargs["method"]=m
      if(best_result is not None):
        ## already did one iteration..
        ## give new initialization vector
        new_kwargs["init"]=best_result["pars_vec"]
        

      res=self.optimize_single(**new_kwargs)

      if(best_result is not None):
        if(res["fnval"]<best_result["fnval"]):
          best_result=res
      else: 
        best_result=res


    ### FIXME should probably merge results and return combined thing

    if(self.callback is not None):

      self.callback(*self.callback_args, force=True)
      
    return best_result

    
  #def Optimize(self, method="migrad", maxcalls=15000, printMode=1, strategy=1, tolerance=1.0, scan_tuples=[], learn_gp=False, nlopt_ftol_opt=1e-5,lq_loss="linear"):
  def optimize_single(self, **kwargs):#maxcalls=15000, printMode=1, strategy=1, tolerance=1.0, scan_tuples=[], learn_gp=False, nlopt_ftol_opt=1e-5,lq_loss="linear"):
      print "------------ BEGING OPTIIZE ---------------------"
      print kwargs
      ## a result dictionary which will be filled in a standardized way by all different algorithms
      #if(len(init_pars)==0):
      #  print "initial start values are required as position arguments, also for determination of argument structure!"
      #  exit(-1)

      method="L-BFGS-B"
      if(kwargs.has_key("method")):
        method=kwargs["method"]

      if(method not in self.minim_defs.keys()):
        print "method ", method, " not supported ..."
        print "supported: ", self.minim_defs.keys()
        exit(-1)

      log_steps=True
      if(kwargs.has_key("log_steps")):
        log_steps=kwargs["log_steps"]

      force_no_gradient=False
      if(kwargs.has_key("force_no_gradient")):
        force_no_gradient=kwargs["force_no_gradient"]

      minimizer_kwargs=dict()
      if(kwargs.has_key("minimizer_kwargs")):
        minimizer_kwargs=kwargs["minimizer_kwargs"]


      ## flipping the sign for certain sampling methods, which want to find the maximum, not minimum
      flip_sign=False
      if("sampyl" in method or "emcee" in method):
        flip_sign=True

      init_parvec=None
      if(kwargs.has_key("init")):
        if(type(kwargs["init"])!=list and type(kwargs["init"])!=numpy.ndarray):
          print "require list as initialized parameter in optimize!"
          exit(-1)

        init_parvec=kwargs["init"]

      results=dict()

      self.results=dict()

      ## see if weant to constrain parameters during this minimzation
      this_param_constraints=self.generate_constraints_list(**kwargs)

      minim_mask=numpy.zeros(len(self.init_parameter_values_single_vec))==0
        #### do an additional modification if we want to fix parameters / profile parameters out....

      ## get a single-vectorized argument function
      minim_functions=self.get_wrapped_fn(argument_type="single", force_no_gradient=force_no_gradient, log_steps=log_steps, flip_sign=flip_sign)


      if(len(this_param_constraints)>1):

        ## we do have further constraints (fixing / profiling parameters .... create a new functions that respects these constraints here)

        minim_functions, minim_mask=self.get_constrained_functions(minim_functions, this_param_constraints, argument_type="single")
        
      else:
       
        minim_functions=[minim_functions]

      
      ## if no initial parvec is given, start with the real initialized parameters
      if(init_parvec==None):
        init_parvec=self.init_parameter_values_single_vec[minim_mask]

      """
      if(self.minim_defs[method][1]=="multi"):

        for ind in range(len(minim_functions)):
          minim_functions[ind]=self.make_multi(minim_functions[ind])
      """

      logged_pars=[]

      ## SCIPY

      # holds best values if logging is off
      best_llh=None
      best_vals=None
      best_valdict=None

      if(self.minim_defs[method][0]=="scipy"):

        if not MSCIPY:
          print "Opimization with %s requires scipy .. scipy not found..." % method
          exit(-1)

        scipy_bounds=[]
        for b in self.bounds.keys():

          if(b in self.param_names[minim_mask]):
            scipy_bounds.append(self.bounds[b])
        
        self.is_optimizing=True
        tbef=time.time()

        for scipy_minim_function in minim_functions:

          scipy_opts=dict()#copy.copy(kwargs)
          scipy_opts["method"]=method
          scipy_opts["bounds"]=scipy_bounds

          scipy_minim_function_jac=None

          if(force_no_gradient == False):
            scipy_minim_function_jac=jacobian(scipy_minim_function)
            scipy_opts["jac"]=scipy_minim_function_jac

          if(minimizer_kwargs.has_key("niter")):
            minimizer_kwargs["maxiter"]=minimizer_kwargs["niter"]
            del minimizer_kwargs["niter"]
            
          scipy_opts["options"]=minimizer_kwargs
          #for minim_kwarg in minimizer_kwargs.keys():
          #  scipy_opts[minim_kwarg]=minimizer_kwargs[minim_kwarg]

         
          
          res=scipy.optimize.minimize(scipy_minim_function, init_parvec, **scipy_opts)

          ## not logging .. so we must see after each minimization which is the best result ... and save it
          if(not log_steps):
            update_best=False
            if(best_llh is None):
              update_best=True
            else:
              if(res["fun"]<best_llh):
                update_best=True
            if(update_best):
              
              best_llh=res["fun"]
              best_valdict=dict()
              for index, p in enumerate(self.param_names[minim_mask]):
                best_valdict[p]=res["x"][index]
              best_vals=res["x"]

      elif(self.minim_defs[method][0]=="minuit"):

        if not MIMINUIT:
          print "Opimization with %s requires iminuit .. iminuit not found..." % method
          exit(-1)

        minuit_opts=dict()

        for b in self.bounds.keys():

          if(b in self.param_names[minim_mask]):
            minuit_opts["limit_%s" % b]=self.bounds[b]
            minuit_opts["error_%s" % b]=1.0

        minuit_opts["forced_parameters"]=self.param_names[minim_mask]

        migrad_opts=minimizer_kwargs
        if(migrad_opts.has_key("niter")):
            migrad_opts["ncall"]=migrad_opts["niter"]
            del migrad_opts["niter"]
        

        for ind in range(len(self.param_names[minim_mask])):
          minuit_opts[self.param_names[minim_mask][ind]]=init_parvec[ind]


        self.is_optimizing=True
        tbef=time.time()

        for minuit_minim_function in minim_functions:

          if(force_no_gradient == False):

            minuit_jac_fn=self.make_multi(jacobian(minuit_minim_function))
            minuit_opts["grad_fcn"]=minuit_jac_fn

          #if(force_no_gradient == False):
          #  minuit_minim_function_jac=jacobian(minuit_minim_function)
          #  minuit_opts["grad_fcn"]=minuit_minim_function_jac

         
          mobject=iminuit.Minuit(self.make_multi(minuit_minim_function), **minuit_opts)



          res=None
          if(method=="migrad"):
            res=mobject.migrad(**migrad_opts)
          elif(method=="simplex"):
            res=mobject.simplex()

          par_results=mobject.get_param_states()
          
          ## not logging .. so we must see after each minimization which is the best result ... and save it
          if(not log_steps):
            update_best=False
            if(best_llh is None):
              update_best=True
            else:
              if(res["fval"]<best_llh):
                update_best=True
            if(update_best):
              
              best_llh=res["fval"]

              best_valdict=dict()
              for index, p in enumerate(self.param_names[minim_mask]):
                best_valdict[p]=res["x"][index]
              best_vals=res["x"]

      elif(self.minim_defs[method][0]=="sampyl"):

        if not MSAMPYL:
          print "Sampling with %s requires sampyl .. sampyl not found..." % method
          exit(-1)

        sampyl_opts=dict()

        #for b in self.bounds.keys():

        #  if(b in self.param_names[minim_mask]):
        #    minuit_opts["limit_%s" % b]=self.bounds[b]
        #    minuit_opts["error_%s" % b]=1.0

        sampyl_opts=minimizer_kwargs

        if(not sampyl_opts.has_key("niter")):
          print "need number of iterations for Sampyl Sampling!!!!"

        niter=sampyl_opts["niter"]
        del sampyl_opts["niter"]


        self.is_optimizing=True
        tbef=time.time()

        for sampyl_minim_function in minim_functions:

          collections.OrderedDict()

          
          chain=0.0
          if(method=="sampyl_metropolis"):
            metro = sampyl.Metropolis(sampyl_minim_function, {'x': init_parvec})
            chain = metro.sample(niter, **sampyl_opts)
          elif(method=="sampyl_nuts"):
            nuts = sampyl.NUTS(sampyl_minim_function, {'x': init_parvec})
            chain = nuts.sample(niter, **sampyl_opts)
          else:
            print "uknown algorithm ", method
            exit(-1)
 
          ## not logging .. so we must see after each minimization which is the best result ... and save it
          if(not log_steps):
            update_best=False
            if(best_llh is None):
              update_best=True
            else:
              if(res["fval"]<best_llh):
                update_best=True
            if(update_best):
              
              best_llh=res["fval"]

              best_valdict=dict()
              for index, p in enumerate(self.param_names[minim_mask]):
                best_valdict[p]=res["x"][index]
              best_vals=res["x"]

      elif(self.minim_defs[method][0]=="emcee"):

        nwalkers=4*len(self.param_names[minim_mask])
        minimizer_kwargs
        if(not minimizer_kwargs.has_key("nwalkers")):
            minimizer_kwargs["nwalkers"]=nwalkers
        else:
            nwalkers=minimizer_kwargs["nwalkers"]

        if(not minimizer_kwargs.has_key("niter")):
            print "require *niter* kw for emcee to define num of samples niter/nwalkers"
            exit(-1)

          #print numpy.fabs(0.2)
          #init_sigma_perturb=numpy.fabs(0.2*numpy.array(self.init_param_values))
          #init_sigma_perturb=numpy.where(init_sigma_perturb==0, 0.1, init_sigma_perturb)
        tbef=time.time()
        self.is_optimizing=True
        for mfunction in minim_functions:

          inv_hesse=numpy.eye(len(init_parvec))*1e-6
          """
          if(MAUTOGRAD):
            print "trying to calculate hesse..."
            hesse=jacobian(jacobian(mfunction))(init_parvec)
            inv_hesse=scipy.linalg.pinvh(hesse)
          """
          
          random_walker_init=numpy.random.multivariate_normal(init_parvec, inv_hesse, nwalkers-1).tolist()
          


          random_walker_init.append(init_parvec.tolist())
          
          sampler = emcee.EnsembleSampler(nwalkers, len(init_parvec), mfunction)

          #print pos, prob
          sec_pos, sec_prob, sec_state=sampler.run_mcmc(numpy.array(random_walker_init), int( float(minimizer_kwargs["niter"])/float(nwalkers)))


      ## after the optimization is done .. save stuff
      self.is_optimizing=False
      total_optimization_time=time.time()-tbef

      result_object=dict()
      result_object["method"]=method
      result_object["niter"]=copy.copy(self.cur_niter)
      result_object["total_time"]=total_optimization_time
      result_object["time_per_iter"]=total_optimization_time/float(self.cur_niter)
      


      if(log_steps):
        parsteps=numpy.array(self.cur_step_params).T
        steps=dict()

        for index, p in enumerate(self.param_names):
          steps[p]=parsteps[index]

        result_object["par_steps"]=steps
        result_object["fnval_steps"]=numpy.array(self.cur_step_fnvals)

        best_mask=result_object["fnval_steps"]==min(result_object["fnval_steps"])
        result_object["fnval"]=result_object["fnval_steps"][best_mask][0]

        best_dict=collections.OrderedDict()
        best_pars=[]
        for index, p in enumerate(self.param_names[minim_mask]):
          best_dict[p]=steps[p][best_mask][0]
          best_pars.append(best_dict[p])

        print "minim mask...", minim_mask
        best_pars=numpy.array(best_pars)

        result_object["pars_dict"]=best_dict
        result_object["pars_vec"]=best_pars

        self.total_parvec.extend(self.cur_step_params)
        self.total_fnvals.extend(self.cur_step_fnvals)

      else:
          
        ## update results from the best iteration

        result_object["pars_dict"]=best_valdict
        result_object["pars_vec"]=best_vals
        result_object["fnval"]=best_llh

      
      #if(scipy_minim_function_jac is not None):
      #  result_object["bestfit_jac"]=scipy_minim_function_jac(result_object["pars_vec"])
        

      ### calculate hessian at best fit point if possible

      """
      if(scipy_minim_function_hess is not None):
        def is_pos_def(x):
            return numpy.all(numpy.linalg.eigvals(x) > 0)

        hessian=scipy_minim_function_hess(result_object["pars_vec"])

        print hessian.shape

        print hessian
        inv_hessian=scipy.linalg.pinvh(hessian)
        
        result_object["inv_hessian"]=inv_hessian
        result_object["posdef"]=is_pos_def(inv_hessian)
      """

      
      
      self.optimization_results.append(result_object)


      self.cur_niter=0
      self.cur_step_fnvals=[]
      self.cur_step_params=[]

      return result_object

      """
      for index, n in enumerate(self.minim_function.func_code.co_varnames):
        self.results["values"][n]=res["x"][index]
      results["fval"]=res["fun"]
      
      

      elif(method=="BOBYQA" or "DIRECT" in method):

        init_values=[self.m_kwargs[i] for i in self.param_names]

        bounds=numpy.array([self.m_kwargs["limit_%s" %i] for i in self.param_names])
        init_steps=[self.m_kwargs["error_%s" %i] for i in self.param_names]
        def wrapper_fn(x, grad=[]):

          res=self.minim_function(*x)
          if(learn_gp):
            if(self.model_mcmc == False):

              self.model_gp.add_data(x, -res)
              self.model_gp = reggie.MCMC(self.model_gp, n=10, burn=100)

              self.model_mcmc=True
            else:

              self.model_gp.add_data(x, -res)
         
            ## learn additionally a GP and store the results
            
          return res

        nlopt_par=None

        if(method=="BOBYQA"):
          nlopt_par=nlopt.LN_BOBYQA
        elif(method=="DIRECT"):
          nlopt_par=nlopt.GN_DIRECT
        elif(method=="DIRECT_L"):
          nlopt_par=nlopt.GN_DIRECT_L
        elif(method=="DIRECT_L_NOSCAL"):
          nlopt_par=nlopt.GN_DIRECT_L_NOSCAL
        elif(method=="DIRECT_NOSCAL"):
          nlopt_par=nlopt.GN_DIRECT_NOSCAL
        elif(method=="DIRECT_L_RAND"):
          nlopt_par=nlopt.GN_DIRECT_L_RAND
        elif(method=="DIRECT_L_RAND_NOSCAL"):
          nlopt_par=nlopt.GN_DIRECT_L_RAND_NOSCAL
        elif(method=="ORIG_DIRECT"):
          nlopt_par=nlopt.GN_ORIG_DIRECT
        elif(method=="ORIC_DIRECT_L"):
          nlopt_par=nlopt.GN_ORIG_DIRECT_L

        opt = nlopt.opt(nlopt_par, len(init_values))
        opt.set_maxeval(maxcalls)
        opt.set_lower_bounds(bounds[:,0])
        opt.set_upper_bounds(bounds[:,1])
        print init_steps
        opt.set_initial_step(init_steps)
        opt.set_min_objective(wrapper_fn)
        opt.set_ftol_abs(nlopt_ftol_opt)
        result_values = opt.optimize(init_values)
        fmin = opt.last_optimum_value()

        self.results=dict()
        self.results["success"]=True
        self.results["values"]=dict()

        for i, vname in enumerate(self.minim_function.func_code.co_varnames):
          self.results["values"][vname]=result_values[i]
        self.results["fval"]=fmin

      elif(method=="TNC" or method=="COBYLA" or method =="SLSQP"):
        init_values=[self.m_kwargs[i] for i in self.minim_function.func_code.co_varnames]

        bounds=[self.m_kwargs["limit_%s" %i] for i in self.minim_function.func_code.co_varnames]

        def wrapper_fn(x):

          res=self.minim_function(*x)
          if(learn_gp):
            if(self.model_mcmc == False):

              self.model_gp.add_data(x, -res)
              self.model_gp = reggie.MCMC(self.model_gp, n=10, burn=100)

              self.model_mcmc=True
            else:

              self.model_gp.add_data(x, -res)
         
            ## learn additionally a GP and store the results
            
          return res
      elif(method=="least_squares"):
        ## calls scipy.least_squares .. function must be residual
        init_values=[self.m_kwargs[i] for i in self.minim_function.func_code.co_varnames]
        self.m_kwargs.has_key("loss")
        def wrapper_fn(x):
          return self.minim_function(*x)

        
        print opti.__version__
        lq_result=opti.least_squares(wrapper_fn, init_values, loss=lq_loss,max_nfev=maxcalls)

        print lq_result

        exit(-1)
      elif(method=="pybo"):

        def wrapper_fn(x):
          return -self.minim_function(*x)

        if(maxcalls == 15000):
          maxcalls=30

        bounds=[self.m_kwargs["limit_%s" %i] for i in self.minim_function.func_code.co_varnames]
        if(len(bounds)!=len(self.minim_function.func_code.co_varnames)):
          print "---------------- ERROR - PYBO needs bounds on all parameters!!!!"
          exit(-1)
        best, model, info = pybo.solve_bayesopt(wrapper_fn, bounds, niter=maxcalls, verbose=True)

        self.results=dict()
        self.results["success"]=True
        self.results["values"]=dict()
        self.results["minim_history"]=info
        self.results["gp_model"]=model
        self.results["fval"]=wrapper_fn(best)
        self.results["ncalls"]=maxcalls

        for index, n in enumerate(self.minim_function.func_code.co_varnames):
          self.results["values"][n]=best[index]

      elif( (method=="migrad") or (method =="simplex")): ### USE MINUIT
 
        args=self.minim_function.func_code.co_varnames

        #var_string_wo=",".join([a.replace("_", "") for a in args])
        var_string=",".join(args)
        list_string="["+var_string+"]"

        
        dict_str="{%s=}"


        #test_args=dict(d_azi=0, d_zen=0, error_d_azi=0.5,error_d_zen=0.4, limit_d_zen=(-4,4), limit_d_azi=(-2,2))
        self.m_kwargs["errordef"]=0.5
        self.m_kwargs["print_level"]=printMode

        m_object=minuit.Minuit(retllh, **self.m_kwargs)
         
        #print "the fitarg"
        #print m_object.fitarg
        #exit(-1)

        #m_object.set_up(0.5)
        
        #m_object.set_print_level(printMode)
        #m_object.maxcalls=maxcalls
        
        m_object.set_strategy(strategy)
        m_object.tol=tolerance


        minim_success=True       
        errmsg=""
        
        
        return_struct=None
        try:
          if(method=="migrad"):
            return_struct= m_object.migrad(ncall=maxcalls)
          elif(method=="simplex"):
            m_object.simplex()
          elif(method=="scan"):
            if(len(scan_tuples)==0):
              print "error - need (param, bins, low, high) for scanning a dimension with minuit, inside a 4 tuple"
              exit(-1)
            m_object.scan(*scan_tuples)
        
        except:
          a=sys.exc_info() 
          print "Unexpected error in minimization:", a[0]
          errmsg=a[1].__str__()
          minim_success=False
         
        self.results=dict()
        
        self.results["success"]=minim_success
        self.results["errmsg"]=errmsg
        self.results["edm"]=m_object.edm
        self.results["values"]=m_object.values
        self.results["args"]=m_object.args
        self.results["errors"]=m_object.errors
        self.results["cov"]=m_object.covariance
        self.results["fval"]=m_object.fval
        self.results["ncalls"]=m_object.ncalls
        self.results["returnstruct"]=return_struct

        print return_struct

        self.results["has_accurate_covariance"]=return_struct[0]["has_accurate_covar"]
        self.results["has_covariance"]=return_struct[0]["has_covariance"]

        self.m_object=m_object
      else:
        print "error - unknown method ", method

      if(learn_gp):

        self.results["gp_model"]=self.model_gp

    else:
      print "error - function is none - cannot minimize!"  
      """
  
  def parscan(self, scan_param_names, range=dict(), fix=dict(), profile=dict(), pts_per_dim=10,grad_pts_per_dim=None, filename=None, profile_method="L-BFGS-B", draw_grads=False,show_minim_steps=False, show_deltallh=[], true_val=None, ax=None, color="black", set_relative_norm=None, ylabel=r"-$2\Delta$-LLH", mult_factor=1.0, log=True,ylim=None):
    
    if(len(scan_param_names) > 2 or len(scan_param_names) == 0):
      
      print "need at least 1 and max. 2 params for scanning (1d/2d scan plots)"
      return
    
    for p in scan_param_names:
      if p not in self.param_names:
        print p , "is not existing in param names .. scan not possible "
        print "possible parnames are:"
        print self.param_names
        exit(-1)

    for pname in fix.keys():
      if pname not in self.param_names:
        print pname, " does not exist in param names and can not be fixed .. "
        print "possible parnames are:"
        print self.param_names
        exit(-1)
    
    scan_ndim=len(scan_param_names)

    scan_range=[]

    scan_names_real_order=[]
    scan_names_indices=[]


    constraint_keys=dict()

    for fkey in fix.keys():
      constraint_keys["fix_"+fkey]=fix[fkey]

    for pkey in profile.keys():
      constraint_keys["profile_"+pkey]=profile[pkey]

    this_param_constraints=self.generate_constraints_list(**constraint_keys)



    for pind, pname in enumerate(self.param_names):
      if(pname in scan_param_names):
        
        print "determining scanrange pname: ", pname
        scan_names_real_order.append(pname)
        scan_names_indices.append(pind)
        if(pname in range.keys()):

          print "pname in range"
          ## the scan range should be the intersection of the allowed range by bounds and the specified rangein the function
          scan_range.append( [max([range[pname][0], self.bounds[pname][0]]),  min([range[pname][1], self.bounds[pname][1]])]  )

        else:
          if(len(self.optimization_results)>0):
            print "determining scanrange from opt results..."
            scan_range.append([min(self.optimization_results[-1]["par_steps"][pname]), max(self.optimization_results[-1]["par_steps"][pname])  ])

          else:
            print "specify range or at least let one minimization run through to get relevant range!"
            exit(-1)


    minim_function=self.get_wrapped_fn(argument_type="single", log_steps=False)

    minim_mask=numpy.zeros(len(self.init_parameter_values_single_vec))==0
    #### do an additional modification if we want to fix parameters / profile parameters out....

    minim_function_single, minim_mask=self.get_constrained_functions(minim_function, this_param_constraints, argument_type="single")
    minim_function_single=minim_function_single[0]


    if(len(this_param_constraints)>1):

      minim_function, minim_mask=self.get_constrained_functions(minim_function, this_param_constraints, argument_type="multi")
      minim_function=minim_function[0]

    if(set_relative_norm is not None):
      relative_norm_shift=set_relative_norm




    def get_uncertainty_1d(h, delta_llh):

      minimum_mask=h.bin_entries.min()==h.bin_entries

      ymin_index=numpy.where(minimum_mask)[0][0]

      larger_min_mask=numpy.arange(len(minimum_mask)) > ymin_index
      smaller_min_mask=numpy.arange(len(minimum_mask)) < ymin_index




      
      delta_llh_mask=h.bin_entries<(h.bin_entries.min()+delta_llh)
      
      
      min_index=numpy.where( (delta_llh_mask==True) & smaller_min_mask)[0]
      if(len(min_index)>0):
        min_index=min_index[0]
      else:
        min_index=0


      max_index=numpy.where( (delta_llh_mask==True) & larger_min_mask)[0]
      if(len(max_index)>0):
        max_index=max_index[-1]
      else:
        max_index=len(h.edges[0])-2


      min_val=h.edges[0][min_index]
      max_val=h.edges[0][max_index+1]

      
      return min_val, h.edges[0][ymin_index], max_val


    if(scan_ndim==1):


      h=hist.HistoFromFunction( numpy.linspace(scan_range[0][0], scan_range[0][1], pts_per_dim), minim_function, vectorized=False)
      h.array_names=scan_param_names

      relative_norm_shift=0.0
      if(set_relative_norm is not None):
        relative_norm_shift-=h.bin_entries.min()

        h.bin_entries-=h.bin_entries.min()

        h.bin_entries+=set_relative_norm
      h.bin_entries*=mult_factor
      h.update_all_attributes()
      rax=None
      if(ax is not None):
        rax=h.plot(return_ax=True, ax=ax, normalplotting=True, color=color)
      else:
        rax=h.plot(return_ax=True, normalplotting=True,color=color)

      min_max=[min(h.bin_entries)-1, max(h.bin_entries)+5]

      if(draw_grads):

        ## numerical gradient
        h_diff=h.diff()
        h_diff.plot(ax=rax,color="blue", linestyle="--", normalplotting=True)

        if(h_diff.bin_entries.min()<min_max[0]):
          min_max[0]=h_diff.bin_entries.min()-1
        if(h_diff.bin_entries.max()>min_max[1]):
          min_max[1]=h_diff.bin_entries.max()-5

        ## analytical gradient
        jac_fn=jacobian(minim_function_single)

        true_grad=[]
        for xv in numpy.linspace(scan_range[0][0], scan_range[0][1], pts_per_dim):
          
          true_grad.append(jac_fn(numpy.array([xv]))[0])

        if(min(true_grad)<min_max[0]):
          min_max[0]=min(true_grad)-1
        if(max(true_grad)>min_max[1]):
          min_max[1]=max(true_grad)-5

        rax.plot(numpy.linspace(scan_range[0][0], scan_range[0][1], pts_per_dim), true_grad, color="red")
        

      if(show_minim_steps):

        ns=len(self.optimization_results[-1]["par_steps"][scan_param_names[0]])
        for ind in numpy.arange(ns):
          this_color=pylab.cm.gray( (float(ind)/float(ns) ))
          rax.plot(self.optimization_results[-1]["par_steps"][scan_param_names[0]][ind], self.optimization_results[-1]["fnval_steps"][ind]+relative_norm_shift, "o",color=this_color)
        rax.plot(self.optimization_results[-1]["par_steps"][scan_param_names[0]][-1:], self.optimization_results[-1]["fnval_steps"][-1:]+relative_norm_shift, "o",color="red")

        #rax.plot(self.optimization_results[-1]["par_steps"][scan_param_names[0]], self.optimization_results[-1]["fnval_steps"], "o-",color="black")
        #rax.plot(self.optimization_results[-1]["par_steps"][scan_param_names[0]][-1:], self.optimization_results[-1]["fnval_steps"][-1:], "o",color="red")

      
      if( len(show_deltallh) > 0 ):
        for delta_llh in show_deltallh:
          minval,midval,maxval=get_uncertainty_1d(h, delta_llh)
          
          err_low=midval-minval
          err_high=maxval-midval

          
          rax.axvline(minval, color=color, label=r"$\Delta$LLH: $%.3f^{-%.3f}_{+%.3f}$" % (midval, err_low, err_high))
          rax.axvline(maxval, color=color)

          

      if(true_val is not None):
        rax.axvline(true_val, color="black", label="Ground Truth")
      rax.set_xlim( [scan_range[0][0],scan_range[0][1] ])
      rax.set_ylim(min_max)
      if(ylim!=None):
        rax.set_ylim(ylim)

      #ax.set_xlim([-0.02, 0.02])
      #ax.set_ylim([-600, -200])
      rax.legend()
      rax.set_ylabel(ylabel)



      if(filename is not None):
        pylab.savefig(filename)

      return rax, h
      
    elif(scan_ndim==2):

      x_vals=numpy.linspace(scan_range[0][0], scan_range[0][1], pts_per_dim)
      y_vals=numpy.linspace(scan_range[1][0], scan_range[1][1], pts_per_dim)

      print x_vals

      print y_vals

      print scan_range


      
      h=hist.HistoFromFunction2d( x_vals, y_vals,  minim_function)


      if(set_relative_norm is not None):
        print "min binentries", h.bin_entries.min()

        h.bin_entries-=h.bin_entries.min()

        h.bin_entries+=set_relative_norm

      h.array_names=scan_names_real_order
      h.update_all_attributes()

      z_min=h.bin_entries.min()
      z_max=h.bin_entries.max()

      if(ylim is not None):
        z_min=ylim[0]
        z_max=ylim[1]


      rax=h.plot(return_ax=True, colorbar=True, use_grid=True, log=log, zmin=z_min, zmax=z_max)
      
      ## draw contour
      
      
      if len(show_deltallh)>0:
        for deltallh in show_deltallh:
          X, Y = numpy.meshgrid(x_vals, y_vals)
          CS = rax.contour(X, Y, h.bin_entries.T,levels=[ h.bin_entries.min()+deltallh],colors=["white"])
          rax.clabel(CS, [ h.bin_entries.min()+deltallh] , fmt={h.bin_entries.min()+deltallh: "%.1f" % float(deltallh)}, inline=1, color="white", fontsize=10)

      if(show_minim_steps):
        ns=len(self.optimization_results[-1]["par_steps"][scan_param_names[0]])
        for ind in numpy.arange(ns):
          this_color=pylab.cm.gray( (float(ind)/float(ns) ))
          rax.plot(self.optimization_results[-1]["par_steps"][scan_names_real_order[0]][ind], self.optimization_results[-1]["par_steps"][scan_names_real_order[1]][ind], "o",color=this_color)
        rax.plot(self.optimization_results[-1]["par_steps"][scan_names_real_order[0]][-1:], self.optimization_results[-1]["par_steps"][scan_names_real_order[1]][-1:], "o",color="red")

      best_x=self.optimization_results[-1]["par_steps"][scan_names_real_order[0]][-1:]
      best_y=self.optimization_results[-1]["par_steps"][scan_names_real_order[1]][-1:]
      #rax.plot( [best_x, best_x+self.optimization_results[-1]["bestfit_jac"][scan_names_indices[0]]], [best_y, best_y+self.optimization_results[-1]["bestfit_jac"][scan_names_indices[1]]],ls="--", lw=3.0,color="black")


      if(draw_grads):
        jac_fn=jacobian(minim_function_single)

        if(grad_pts_per_dim==None):
          grad_pts_per_dim=pts_per_dim/10

        x_vals_grad=numpy.linspace(scan_range[0][0], scan_range[0][1], grad_pts_per_dim)
        y_vals_grad=numpy.linspace(scan_range[1][0], scan_range[1][1], grad_pts_per_dim)

        xwidth_per_grad=(scan_range[0][1]-scan_range[0][0])/float(grad_pts_per_dim)
        ywidth_per_grad=(scan_range[1][1]-scan_range[1][0])/float(grad_pts_per_dim)

        largest_grad_aspect_ratio=1.0

        graddict=dict()

        for xg in x_vals_grad:
          for yg in y_vals_grad:
            
            grad_res=jac_fn(anumpy.array([xg,yg]))
            print "x,y: ", xg,yg
            print "grad", grad_res


            print "single", minim_function_single(numpy.array([xg,yg]))
            print "multi", minim_function(xg,yg)

            graddict[(xg,yg)]=copy.deepcopy(grad_res)
            #graddict[(xg,yg)]=copy.deepcopy(grad_res[::-1])

            if(graddict[(xg,yg)][0]/xwidth_per_grad > largest_grad_aspect_ratio):
              largest_grad_aspect_ratio=graddict[(xg,yg)][0]/xwidth_per_grad

            if(graddict[(xg,yg)][1]/ywidth_per_grad > largest_grad_aspect_ratio):
              largest_grad_aspect_ratio=graddict[(xg,yg)][1]/ywidth_per_grad
        print "ASPECT", largest_grad_aspect_ratio
        largest_grad_aspect_ratio*=0.8
        for xg in x_vals_grad:
          for yg in y_vals_grad:
            rax.arrow( xg-graddict[(xg,yg)][0]/(largest_grad_aspect_ratio*2.0) , yg-graddict[(xg,yg)][1]/(largest_grad_aspect_ratio*2.0), graddict[(xg,yg)][0]/(largest_grad_aspect_ratio*2.0), graddict[(xg,yg)][1]/(largest_grad_aspect_ratio*2.0) , fc="k", ec="k" , head_width=0.001)
            #rax.plot([xg-graddict[(xg,yg)][0]/(largest_grad_aspect_ratio*2.0), xg+graddict[(xg,yg)][0]/(largest_grad_aspect_ratio*2.0)],[yg-graddict[(xg,yg)][1]/(largest_grad_aspect_ratio*2.0), yg+graddict[(xg,yg)][1]/(largest_grad_aspect_ratio*2.0)] ,ls="--", lw=3.0,color="black")
            #rax.plot([xg],[yg] , marker="x",color="black")
            rax.text(xg, yg, "(%.2e, %.2e)" % (graddict[(xg,yg)][0], graddict[(xg,yg)][1]), size=7)


        


      rax.set_xlim( h.edges[0][0], h.edges[0][-1])
      rax.set_ylim( h.edges[1][0], h.edges[1][-1])

      if(filename is not None):
        pylab.savefig(filename)

      return rax, h
      



    print "done .."


