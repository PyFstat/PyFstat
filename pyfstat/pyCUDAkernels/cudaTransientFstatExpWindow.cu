__global__ void cudaTransientFstatExpWindow ( float *input,
                                              unsigned int numAtoms,
                                              unsigned int TAtom,
                                              unsigned int t0_data,
                                              unsigned int win_t0,
                                              unsigned int win_dt0,
                                              unsigned int win_tau,
                                              unsigned int win_dtau,
                                              unsigned int Fmn_rows,
                                              unsigned int Fmn_cols,
                                              float *Fmn
                                            )
{

  /* match CUDA thread indexing and high-level (t0,tau) indexing */
  unsigned int m         = blockDim.x * blockIdx.x + threadIdx.x; // t0:  row
  unsigned int n         = blockDim.y * blockIdx.y + threadIdx.y; // tau: column
  /* unraveled 1D index for 2D output array */
  unsigned int outidx    = Fmn_cols * m + n;

  /* hardcoded copy from lalpulsar */
  unsigned int TRANSIENT_EXP_EFOLDING = 3;

  if ( (m < Fmn_rows) && (n < Fmn_cols) ) {

    /* compute Fstat-atom index i_t0 in [0, numAtoms) */
    unsigned int TAtomHalf = TAtom/2; // integer division
    unsigned int t0 = win_t0 + m * win_dt0;
    /* integer round: floor(x+0.5) */
    int i_tmp = ( t0 - t0_data + TAtomHalf ) / TAtom;
    if ( i_tmp < 0 ) {
        i_tmp = 0;
    }
    unsigned int i_t0 = (unsigned int)i_tmp;
    if ( i_t0 >= numAtoms ) {
        i_t0 = numAtoms - 1;
    }

    /* translate n into an atoms end-index
     * for this search interval [t0, t0+Tcoh],
     * giving the index range of atoms to sum over
     */
    unsigned int tau = win_tau + n * win_dtau;

    /* get end-time t1 of this transient-window search
     * for given tau, what Tcoh should the exponential window cover?
     * for speed reasons we want to truncate
     * Tcoh = tau * TRANSIENT_EXP_EFOLDING
     * with the e-folding factor chosen such that the window-value
     * is practically negligible after that, where it will be set to 0
     */
//     unsigned int t1 = lround( win_t0 + TRANSIENT_EXP_EFOLDING * win_tau);
    unsigned int t1 = t0 + TRANSIENT_EXP_EFOLDING * tau;

      /* compute window end-time Fstat-atom index i_t1 in [0, numAtoms)
       * using integer round: floor(x+0.5)
       */
    i_tmp = ( t1 - t0_data + TAtomHalf ) / TAtom  - 1;
    if ( i_tmp < 0 ) {
        i_tmp = 0;
    }
    unsigned int i_t1 = (unsigned int)i_tmp;
    if ( i_t1 >= numAtoms ) {
        i_t1 = numAtoms - 1;
    }

    /* now we have two valid atoms-indices [i_t0, i_t1]
     * spanning our Fstat-window to sum over
     */

    float Ad    = 0.0f;
    float Bd    = 0.0f;
    float Cd    = 0.0f;
    float Fa_re = 0.0f;
    float Fa_im = 0.0f;
    float Fb_re = 0.0f;
    float Fb_im = 0.0f;

    unsigned short input_cols = 7; // must match input matrix!

    /* sum up atoms */
    for ( unsigned int i=i_t0; i<=i_t1; i++ ) {

      unsigned int t_i = t0_data + i * TAtom;

      float win_i = 0.0;
      if ( t_i >= t0 && t_i <= t1 ) {
        float x = 1.0 * ( t_i - t0 ) / tau;
        win_i = exp ( -x );
      }

      float win2_i = win_i * win_i;

      Ad    += input[i*input_cols+0] * win2_i; // a2_alpha
      Bd    += input[i*input_cols+1] * win2_i; // b2_alpha
      Cd    += input[i*input_cols+2] * win2_i; // ab_alpha
      Fa_re += input[i*input_cols+3] * win_i; // Fa_alpha_re
      Fa_im += input[i*input_cols+4] * win_i; // Fa_alpha_im
      Fb_re += input[i*input_cols+5] * win_i; // Fb_alpha_re
      Fb_im += input[i*input_cols+6] * win_i; // Fb_alpha_im

    }

    /* get inverse antenna pattern determinant,
     * following safety checks from
     * XLALComputeAntennaPatternSqrtDeterminant()
     * and estimateAntennaPatternConditionNumber()
     */
    float sumAB  = Ad + Bd;
    float diffAB = Ad - Bd;
    float disc   = sqrt ( diffAB*diffAB + 4.0 * Cd*Cd );
    float denom = sumAB - disc;
    float cond = (denom > 0) ? ((sumAB + disc) / denom) : INFINITY;
    float DdInv = 0.0f;
    if ( cond < 1e4 ) {
      DdInv = 1.0 / ( Ad * Bd - Cd * Cd );
    }

    /* matching compute_fstat_from_fa_fb
     * including default fallback = 0.5*E[2F] in noise
     * when DdInv == 0 due to ill-conditionness of M_munu
     */
    float F = 2;
    if ( DdInv > 0 ) {
      F  = DdInv * (   Bd * ( Fa_re*Fa_re + Fa_im*Fa_im )
                     + Ad * ( Fb_re*Fb_re + Fb_im*Fb_im )
                     - 2.0 * Cd * ( Fa_re * Fb_re + Fa_im * Fb_im )
                   );
    }

    /* store result in Fstat-matrix
     * at unraveled index of element {m,n}
     */
    Fmn[outidx] = F;

  } // ( (m < Fmn_rows) && (n < Fmn_cols) )

} // cudaTransientFstatExpWindow()
