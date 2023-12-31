      subroutine neutrino(i,t,rho,a,z,qtot,
     1   qeebrem,qnpb,qplasma,qsynch,qbubble,qpair,qphoto,qbrem_nn,
     2   qmurca_nucl,qbrem_nucl,qmurca_hyp,qbrem_hyp,
     3   qdurca_np,qdurca_lap,qdurca_smn,qdurca_smla,qdurca_sms0,
     4   qfast,
     5   qdurca_q,qmurca_q,
     6   qpbf_n1s0,qpbf_n3p2,qpbf_p1s0,qpbf_q,
     7   debug,naa)

c **** checked  august 21 1991 **************

      implicit real*8 (a-h,k-z)
      INCLUDE 'size.inc.f'
      INCLUDE 'rho_limits.inc.f'
      INCLUDE 'control_nu.inc.f'
      INCLUDE 'pairing.inc.f'
      INCLUDE 'fermi.inc.f'
      INCLUDE 'profile_comp.inc.f'
      INCLUDE 'profile_star.inc.f'
      INCLUDE 'mag_field.inc.f'

      if (debug.ge.2.) print *,'Entering subroutine `neutrino'' ',
     2       ' T, rho, A, Z = ',t,rho,a,z
c *** ELECTRON-ELECTRON PAIR BREMSSTRAHLUNG:
      if (rho.lt.rhocore) then
         mu_el=kfe(i)*197.
         call neebrem(T,mu_el,qeebrem)
      else
         qeebrem=0.0d0
      end if
c *** ELECTRON-ION PAIR BREMSSTRAHLUNG:
      if (inu_eion.eq.1) then
         if (rho.lt.rhocore) then
            call npb_new(t,rho,qnpb)
         else
            qnpb=0.0d0
         end if
      else if (inu_eion.eq.2) then
         if (rho.lt.rhocore) then
            call npb(t,rho,a,z,qnpb)
         else
            qnpb=0.0d0
         end if
      else
         if (rho.lt.rhocore) then
            qnpb=0.0d0
            if (print_it.ne.1.) then
             print *,'No npb: Rho, Qnpb=',rho,qnpb
             print_it=1.
            end if
         else
            qnpb=0.0d0
         end if
      end if
c *** PLASMA NEUTRINO:
      if (inu_plasma.eq.1) then
         if (rho.lt.rhocore) then
            call nplasma(t,rho,a,z,qplasma)
         else
            qplasma=0.0d0
         end if
      else if (inu_plasma.eq.-1) then
         if (rho.lt.rhocore) then
            call nplasma_old(t,rho,a,z,qplasma)
         else
            qplasma=0.0d0
         end if
      else
         qplasma=0.0d0
      end if
c *** SYNCHROTRON NEUTRINO:
      if (inu_synch.eq.1) then
         if (rho.lt.rhocore) then
            call nsynch(t,nbfield2(i),kfe(i),qsynch)
         end if
      else
         qsynch=0.0d0
      end if
c *** BUBBLE NEUTRINO:
      if (inu_bubble.eq.1) then
         if (rho.lt.rhocore) then
            call nbub(i,t,rho,a,z,qbubble)
         else
            qbubble=0.0d0
         end if
      else
         qbubble=0.0d0
      end if
c *** NEUTRINO PAIR:
      if (inu_pair.eq.1) then
         if (rho.lt.rhocore) then
            call npair(t,rho,a,z,qpair)
         else
            qpair=0.0d0
         end if
      else
         qpair=0.0d0
      end if
c **** PHOTO-NEUTRINO:
      if (inu_photo.eq.1) then
         if (rho.lt.rhocore) then
            call nphoto(t,rho,a,z,qphoto)
         else
            qphoto=0.0d0
         end if
      else
          qphoto=0.0d0
      end if
c *** NN-BREMSTRAHLUNG in the inner crust:
      if ((rho.lt.rhocore).and.(rho.ge.rhodrip)) then
         call nubrem_crust_nn(i,t,v_ion(i),qbrem_nn)
      else
         qbrem_nn=0.d0
      end if
c *** URCA et al. PROCESSES:
      if (rho.ge.rhocore) then
         if (istrange.eq.0) then
            call numurca_nucl(i,t,qmurca_nucl)
            qmurca_nucl=qmurca_nucl*(1.d0+murca_increase)
            qmurca_nucl=qmurca_nucl*fhad(i)
            call nubrem_nucl(i,t,qbrem_nucl)
            qbrem_nucl=qbrem_nucl*(1.d0+murca_increase)
            qbrem_nucl=qbrem_nucl*fhad(i)
            call numurca_hyp(i,t,qmurca_hyp)
            qmurca_hyp=qmurca_hyp*fhad(i)
            call nubrem_hyp(i,t,qbrem_hyp)
            qbrem_hyp=qbrem_hyp*fhad(i)
            if (inu_durca.eq.1) then
             call nudurca_h(i,t,rho,qdurca_np,qdurca_lap,
     1                        qdurca_smn,qdurca_smla,qdurca_sms0)
             qdurca_np=qdurca_np*fhad(i)
             qdurca_lap=qdurca_lap*fhad(i)
             qdurca_smn=qdurca_smn*fhad(i)
             qdurca_smla=qdurca_smla*fhad(i)
             qdurca_sms0=qdurca_sms0*fhad(i)
            else
             qdurca_np=0.0d0
             qdurca_lap=0.0d0
             qdurca_smn=0.0d0
             qdurca_smla=0.0d0
             qdurca_sms0=0.0d0
            end if
c *** FAST neutrino emission:
            call nufast (i,t,rho,qfast)
            qfast=qfast*fhad(i)
c *** QUARK processes:
            call nudurca_q(i,t,rho,qdurca_q)
            call numurca_q(i,t,rho,qmurca_q)
            qdurca_q=qdurca_q*(1.d0-fhad(i))
            qmurca_q=qmurca_q*(1.d0-fhad(i))
            qstrange=0.d0
c *** STRANGE QUARK MATTER processes:
         else if (istrange.eq.1) then
            if (c_nu_str.le.1.d0) then
               call nu_strange(i,T,qstrange,debug)
            else
               qstrange=c_nu_str*(T/1.d9)**p_nu_str
            end if
            qmurca_nucl=0.0d0
            qbrem_nucl=0.0d0
            qmurca_hyp=0.0d0
            qbrem_hyp=0.0d0
            qdurca_np=0.0d0
            qdurca_lap=0.0d0
            qdurca_smn=0.0d0
            qdurca_smla=0.0d0
            qdurca_sms0=0.0d0
            qfast=0.0d0
            qdurca_q=0.0d0
            qmurca_q=0.0d0
         else
            print *,'neutrino: istrange not defined !'
            pause
         end if
      else
         qmurca_nucl=0.0d0
         qbrem_nucl=0.0d0
         qmurca_hyp=0.0d0
         qbrem_hyp=0.0d0
         qdurca_np=0.0d0
         qdurca_lap=0.0d0
         qdurca_smn=0.0d0
         qdurca_smla=0.0d0
         qdurca_sms0=0.0d0
         qfast=0.0d0
         qdurca_q=0.0d0
         qmurca_q=0.0d0
         qstrange=0.d0
      end if
c *** PBF PROCESSES:
      if (istrange.eq.0) then
c      Neutrons 1S0:
       if ((inu_n1s0_pbf.eq.1).and.(i.gt.isf)) then
          call nu_1s0_pbf(t,tcn(i),mstn(i),kfn(i),qpbf_n1s0)
          qpbf_n1s0=qpbf_n1s0*fhad(i)
       else
          qpbf_n1s0=0.0d0
       end if
c      Neutron 3P2:
       if ((inu_n3p2_pbf.eq.1).and.(i.le.isf)) then
          call nu_n3p2_B_pbf(t,tcn(i),mstn(i),kfn(i),qpbf_n3p2)
          qpbf_n3p2=qpbf_n3p2*fhad(i)
       else
          qpbf_n3p2=0.0d0
       end if
c      Protons:
       if (inu_p_pbf.eq.1) then
          call nu_1s0_pbf(t,tcp(i),mstp(i),kfp(i),qpbf_p1s0)
          qpbf_p1s0=qpbf_p1s0*fhad(i)
       else
          qpbf_p1s0=0.0d0
       end if
c      Quarks: TO BE INCLUDED !!!!!!!!
       qpbf_q=0.0d0
       qpbf_q=qpbf_q*(1.d0-fhad(i))
      else
       qpbf_n1s0=0.0d0
       qpbf_n3p2=0.0d0
       qpbf_p1s0=0.0d0
       qpbf_q=0.0d0
      end if
c *** ADDING EVERYTHING:
      qtot=
     1   qeebrem+qnpb+qplasma+qsynch+qbubble+qpair+qphoto+qbrem_nn+
     2   qmurca_nucl+qbrem_nucl+qmurca_hyp+qbrem_hyp+
     3   qdurca_np+qdurca_lap+qdurca_smn+qdurca_smla+qdurca_sms0+
     4   qfast+
     5   qdurca_q+qmurca_q+qstrange+
     6   qpbf_n1s0+qpbf_n3p2+qpbf_p1s0+qpbf_q
c*****
      if (debug.ge.2.) print *,'Exiting subroutine `neutrino'' '
      return

      end







